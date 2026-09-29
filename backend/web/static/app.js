// Browser-side behaviour for the server-rendered pages. Everything the pages show
// is rendered by FastAPI and Jinja; this file only handles what must run in the
// browser: theme, tooltips, the upload queue, the simulation stream, row
// selection and slide navigation. HTMX handles table filters and row loading.

const reduceMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches

// Scrolls an element into view, without animation when reduced motion is on.
function reveal(element) {
  element?.scrollIntoView({ behavior: reduceMotion() ? "auto" : "smooth", block: "start" })
}

// Inserts server-rendered HTML into a container and lets HTMX wire up its attributes.
function mount(container, html) {
  container.innerHTML = html
  if (window.htmx) window.htmx.process(container)
}

// ------------------------------------------------------------------ theme

// Toggles between the dark and light themes and remembers the choice.
function setupTheme() {
  document.querySelectorAll("[data-theme-toggle]").forEach(button => {
    button.addEventListener("click", () => {
      const next = document.documentElement.dataset.theme === "light" ? "dark" : "light"
      document.documentElement.dataset.theme = next
      document.querySelector('meta[name="theme-color"]')?.setAttribute("content", next === "light" ? "#eef2fb" : "#050914")
      try {
        localStorage.setItem("flowlens-theme", next)
      } catch {
        // Storage can be unavailable (private mode); the toggle still works for this visit.
      }
    })
  })
}

// --------------------------------------------------------------- tooltips

// One shared tooltip for every chart mark with a data-tip attribute, shown on
// hover and on keyboard focus.
function setupTooltips() {
  const tooltip = document.querySelector(".tooltip")
  if (!tooltip) return

  // Fills and positions the tooltip for a mark at the given viewport point.
  const show = (mark, x, y) => {
    let tip
    try {
      tip = JSON.parse(mark.dataset.tip)
    } catch {
      return
    }

    tooltip.replaceChildren()
    if (tip.title) {
      const title = document.createElement("strong")
      title.className = "tooltip-title"
      title.textContent = tip.title
      tooltip.append(title)
    }
    for (const row of tip.rows || []) {
      const line = document.createElement("span")
      line.className = "tooltip-row"
      if (row.color) {
        const swatch = document.createElement("i")
        swatch.style.background = row.color
        line.append(swatch)
      }
      const label = document.createElement("span")
      label.textContent = row.label
      const value = document.createElement("b")
      value.textContent = row.value
      line.append(label, value)
      tooltip.append(line)
    }

    tooltip.style.left = `${x}px`
    tooltip.style.top = `${y}px`
    tooltip.hidden = false
  }

  const hide = () => {
    tooltip.hidden = true
  }

  document.addEventListener("mousemove", event => {
    const mark = event.target.closest?.("[data-tip]")
    if (mark) show(mark, event.clientX, event.clientY)
    else hide()
  })
  document.addEventListener("focusin", event => {
    const mark = event.target.closest?.("[data-tip]")
    if (!mark) return hide()
    const box = mark.getBoundingClientRect()
    show(mark, box.left + box.width / 2, box.top)
  })
  document.addEventListener("focusout", hide)
  document.addEventListener("scroll", hide, { passive: true })
}

// --------------------------------------------------------- results views

// Row selection and the flow breakdown: marks the clicked row, scrolls to the
// breakdown once HTMX has loaded it, and handles the Close button.
function setupResults() {
  let selectedPath = null

  document.body.addEventListener("htmx:beforeRequest", event => {
    const row = event.detail.elt
    if (!row.matches?.("tr.row")) return
    selectedPath = row.getAttribute("hx-get")
    row.closest("tbody").querySelectorAll("tr.row").forEach(other => other.setAttribute("aria-selected", other === row ? "true" : "false"))
  })

  document.body.addEventListener("htmx:afterSwap", event => {
    const target = event.detail.target
    if (target.id === "flow-detail" && event.detail.requestConfig?.elt?.matches?.("tr.row")) {
      reveal(target.querySelector(".detail"))
    }
    // Keep the chosen row highlighted after a filter reloads the table.
    if (target.id === "flow-table" && selectedPath) {
      target.querySelectorAll("tr.row").forEach(row => row.setAttribute("aria-selected", row.getAttribute("hx-get") === selectedPath ? "true" : "false"))
    }
  })

  document.body.addEventListener("click", event => {
    const close = event.target.closest("[data-close-detail]")
    if (!close) return
    const holder = close.closest("#flow-detail")
    holder.replaceChildren()
    document.querySelectorAll("tr.row[aria-selected='true']").forEach(row => row.setAttribute("aria-selected", "false"))
    selectedPath = null
  })
}

// ------------------------------------------------------------- simulation

// Dashboard simulation: reads the chosen scenario and speed, opens the
// server-sent event stream and places each rendered fragment on the page.
function setupSimulation() {
  const section = document.querySelector("#simulation")
  if (!section) return

  const $ = selector => section.querySelector(selector)
  const startButtons = document.querySelectorAll("[data-sim-start]")
  const stopButton = $("[data-sim-stop]")
  const resetButton = $("[data-sim-reset]")
  const label = $("[data-sim-label]")
  const results = document.querySelector("#sim-results")
  const initialNow = $("[data-sim-now]").innerHTML
  const initialProgress = $("[data-sim-progress]").innerHTML
  const initialScores = $("[data-sim-scores]").innerHTML
  const choice = name => section.querySelector(`[data-choice="${name}"] [aria-pressed="true"]`).value
  let source = null
  let runId = null

  // Shows which prepared flows the chosen scenario will run.
  const updatePreview = () => {
    const scenario = choice("scenario")
    let count = 0
    $("[data-sim-preview]").querySelectorAll("[data-group]").forEach(item => {
      const shown = scenario === "all" || item.dataset.group === scenario
      item.hidden = !shown
      count += shown
    })
    label.textContent = runId ? "Run again" : `Run ${count} flows`
  }

  section.querySelectorAll("[data-choice]").forEach(group => {
    group.addEventListener("click", event => {
      const button = event.target.closest("button")
      if (!button || button.disabled) return
      group.querySelectorAll("button").forEach(other => other.setAttribute("aria-pressed", other === button ? "true" : "false"))
      if (group.dataset.choice === "scenario") updatePreview()
    })
  })

  // Enables or disables the controls while a run is in progress.
  const setRunning = running => {
    section.classList.toggle("is-running", running)
    $(".progress").hidden = !running
    startButtons.forEach(button => (button.disabled = running))
    section.querySelectorAll("[data-choice] button").forEach(button => (button.disabled = running))
    stopButton.hidden = !running
    $("[data-sim-start]").hidden = running
    resetButton.hidden = running || !runId
  }

  // Loads the full results and insights for the finished (or stopped) run.
  const loadResults = async () => {
    const response = await fetch(`/runs/${runId}/results`)
    if (response.ok) mount(results, await response.text())
  }

  const start = () => {
    if (source) return
    const scenario = choice("scenario")
    const speed = choice("speed")

    runId = null
    results.replaceChildren()
    $("[data-sim-error]").hidden = true
    $("[data-sim-preview]").hidden = true
    $("[data-sim-feed-wrap]").hidden = false
    $("[data-sim-feed-empty]").hidden = false
    $("[data-sim-feed]").replaceChildren()
    $("[data-sim-scores]").innerHTML = initialScores
    setRunning(true)
    reveal(section)

    source = new EventSource(`/simulate?scenario=${encodeURIComponent(scenario)}&speed=${encodeURIComponent(speed)}`)

    source.addEventListener("start", event => {
      const data = JSON.parse(event.data)
      runId = data.run_id
      $("[data-sim-progress]").innerHTML = data.progress
    })

    source.addEventListener("stage", event => {
      $("[data-sim-now]").innerHTML = JSON.parse(event.data).now
    })

    source.addEventListener("flow", event => {
      const data = JSON.parse(event.data)
      $("[data-sim-feed-empty]").hidden = true
      $("[data-sim-feed]").insertAdjacentHTML("afterbegin", data.items)
      $("[data-sim-scores]").innerHTML = data.scores
      $("[data-sim-progress]").innerHTML = data.progress
    })

    source.addEventListener("done", event => {
      const data = JSON.parse(event.data)
      finish()
      $("[data-sim-progress]").innerHTML = data.progress
      $("[data-sim-now]").innerHTML = data.now
      loadResults()
    })

    source.addEventListener("failed", event => {
      finish()
      showError(JSON.parse(event.data).message)
    })

    // A dropped connection before "done" means the server went away.
    source.onerror = () => {
      if (!source) return
      finish()
      showError("Lost the connection to the server. Check that it is still running, then run the simulation again.")
    }
  }

  const showError = message => {
    const error = $("[data-sim-error]")
    error.textContent = message
    error.hidden = false
  }

  // Closes the stream and restores the controls.
  const finish = () => {
    source?.close()
    source = null
    setRunning(false)
    label.textContent = "Run again"
  }

  const stop = async () => {
    finish()
    $("[data-sim-now]").innerHTML = initialNow
    const state = $("[data-sim-progress] .sim-state")
    if (state) {
      state.className = "sim-state sim-state-stopped"
      state.textContent = "Stopped"
    }
    if (!runId) return
    const response = await fetch(`/runs/${runId}/stopped`, { method: "POST" })
    if (response.ok && (await response.json()).flows > 0) loadResults()
  }

  const reset = () => {
    runId = null
    results.replaceChildren()
    $("[data-sim-now]").innerHTML = initialNow
    $("[data-sim-progress]").innerHTML = initialProgress
    $("[data-sim-scores]").innerHTML = initialScores
    $("[data-sim-feed]").replaceChildren()
    $("[data-sim-feed-wrap]").hidden = true
    $("[data-sim-preview]").hidden = false
    $("[data-sim-error]").hidden = true
    resetButton.hidden = true
    updatePreview()
  }

  startButtons.forEach(button => button.addEventListener("click", start))
  stopButton.addEventListener("click", stop)
  resetButton.addEventListener("click", reset)
  window.addEventListener("pagehide", () => source?.close())
  updatePreview()
}

// ----------------------------------------------------------------- upload

// Upload page: collects files from the picker or drag and drop into a queue,
// sends them to /classify/run in one request and shows the results.
function setupUpload() {
  const intake = document.querySelector("[data-intake]")
  if (!intake) return

  const $ = selector => intake.querySelector(selector)
  const drop = $("[data-drop]")
  const input = $("[data-file-input]")
  const list = $("[data-queue-list]")
  const template = document.querySelector("#queue-item")
  const results = document.querySelector("#upload-results")
  const classifyButton = $("[data-classify]")
  const queue = []
  let loading = false

  // Identifies a file by name, size and modification time so it isn't queued twice.
  const keyOf = file => `${file.name}:${file.size}:${file.lastModified}`
  // True for the formats the backend can read.
  const accepted = file => /\.(parquet|csv)$/i.test(file.name)
  // Formats a file size in B, KB or MB.
  const size = bytes => (bytes < 1024 ? `${bytes} B` : bytes < 1048576 ? `${(bytes / 1024).toFixed(1)} KB` : `${(bytes / 1048576).toFixed(1)} MB`)

  // Redraws the queue list, counts and button states.
  const render = () => {
    list.replaceChildren(
      ...queue.map(item => {
        const node = template.content.firstElementChild.cloneNode(true)
        node.querySelector(".queue-name").textContent = item.file.name
        node.querySelector(".queue-name").title = item.file.name
        node.querySelector("[data-size]").textContent = size(item.file.size)
        node.querySelector(".queue-icon").classList.toggle("is-csv", /\.csv$/i.test(item.file.name))
        const status = node.querySelector("[data-status]")
        if (item.report?.error) {
          status.className = "status-error"
          status.textContent = item.report.error
        } else if (item.report) {
          status.className = "status-done"
          status.textContent = `${item.report.rows} ${item.report.rows === 1 ? "flow" : "flows"} read`
        }
        const remove = node.querySelector("[data-remove]")
        remove.setAttribute("aria-label", `Remove ${item.file.name}`)
        remove.disabled = loading
        remove.addEventListener("click", () => {
          queue.splice(queue.indexOf(item), 1)
          render()
        })
        return node
      })
    )

    list.hidden = queue.length === 0
    $("[data-queue-empty]").hidden = queue.length > 0
    $("[data-queue-count]").textContent = queue.length ? `${queue.length} ${queue.length === 1 ? "file" : "files"}` : "No files yet"
    $("[data-classify-label]").textContent = loading ? "Classifying…" : queue.length > 1 ? `Classify ${queue.length} files` : "Classify"
    $("[data-classify-spinner]").hidden = !loading
    classifyButton.disabled = loading || queue.length === 0
    $("[data-clear]").disabled = loading || (queue.length === 0 && !results.childElementCount)
  }

  // Adds picked or dropped files, reporting any that aren't Parquet or CSV.
  const add = fileList => {
    const files = Array.from(fileList || [])
    if (!files.length) return
    const rejected = files.filter(file => !accepted(file)).map(file => file.name)
    const note = $("[data-queue-rejected]")
    note.hidden = rejected.length === 0
    note.textContent = `Not added, because only .parquet and .csv files can be classified: ${rejected.join(", ")}`
    const known = new Set(queue.map(item => item.key))
    files.filter(accepted).filter(file => !known.has(keyOf(file))).forEach(file => queue.push({ key: keyOf(file), file, report: null }))
    render()
  }

  // Sends the queue to the server and shows the results it renders.
  const classify = async () => {
    if (!queue.length || loading) return
    loading = true
    $("[data-classify-error]").hidden = true
    $(".progress").hidden = false
    render()

    const sent = [...queue]
    const form = new FormData()
    sent.forEach(item => form.append("files", item.file))

    try {
      const response = await fetch("/classify/run", { method: "POST", body: form })
      const data = await response.json()
      if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : `The server returned ${response.status}.`)
      sent.forEach((item, index) => (item.report = data.files[index]))
      mount(results, data.html)
    } catch (error) {
      const message = $("[data-classify-error]")
      message.textContent = error instanceof TypeError ? "Couldn't reach the server. Check that it is still running." : error.message
      message.hidden = false
    } finally {
      loading = false
      $(".progress").hidden = true
      render()
    }
  }

  drop.addEventListener("click", event => {
    if (!event.target.closest("[data-pick]")) input.click()
  })
  $("[data-pick]").addEventListener("click", () => input.click())
  input.addEventListener("change", () => {
    add(input.files)
    // Clear the input so choosing the same file again still fires change.
    input.value = ""
  })
  drop.addEventListener("dragover", event => {
    event.preventDefault()
    drop.classList.add("is-dragging")
  })
  drop.addEventListener("dragleave", event => {
    if (!drop.contains(event.relatedTarget)) drop.classList.remove("is-dragging")
  })
  drop.addEventListener("drop", event => {
    event.preventDefault()
    drop.classList.remove("is-dragging")
    add(event.dataTransfer.files)
  })
  classifyButton.addEventListener("click", classify)
  $("[data-clear]").addEventListener("click", () => {
    queue.length = 0
    results.replaceChildren()
    $("[data-queue-rejected]").hidden = true
    $("[data-classify-error]").hidden = true
    render()
  })
  render()
}

// ------------------------------------------------------------ presentation

// Project slides: the side index follows the visible slide, and the floating
// controls and left/right arrow keys move between slides.
function setupDeck() {
  const deck = document.querySelector("[data-deck]")
  if (!deck) return

  const slides = [...deck.querySelectorAll("[data-slide]")]
  const navButtons = [...deck.querySelectorAll("[data-go]")]
  const position = deck.querySelector("[data-deck-position]")
  const [previous, next] = deck.querySelectorAll("[data-step]")
  let active = 0

  // Marks a slide as current in the index and the controls.
  const mark = index => {
    active = index
    navButtons.forEach((button, i) => (i === index ? button.setAttribute("aria-current", "step") : button.removeAttribute("aria-current")))
    position.textContent = `${index + 1} / ${slides.length}`
    previous.disabled = index === 0
    next.disabled = index === slides.length - 1
  }

  const go = index => reveal(slides[Math.max(0, Math.min(slides.length - 1, index))])

  const observer = new IntersectionObserver(
    entries => entries.forEach(entry => entry.isIntersecting && mark(slides.indexOf(entry.target))),
    { rootMargin: "-45% 0px -45% 0px" }
  )
  slides.forEach(slide => observer.observe(slide))

  navButtons.forEach(button => button.addEventListener("click", () => go(Number(button.dataset.go))))
  previous.addEventListener("click", () => go(active - 1))
  next.addEventListener("click", () => go(active + 1))
  window.addEventListener("keydown", event => {
    if (event.target.closest("input, textarea, select, [contenteditable]")) return
    if (event.key === "ArrowRight") go(active + 1)
    if (event.key === "ArrowLeft") go(active - 1)
  })
  mark(0)
}

document.addEventListener("DOMContentLoaded", () => {
  setupTheme()
  setupTooltips()
  setupResults()
  setupSimulation()
  setupUpload()
  setupDeck()
})

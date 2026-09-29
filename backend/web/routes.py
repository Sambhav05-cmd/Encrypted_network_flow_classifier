"""Page routes for the server-rendered interface (FastAPI + Jinja2 + HTMX).

Pages: / (dashboard and simulation), /classify (upload) and /project (slides).
Results are kept in a small in-memory store so table filters, row selection and
CSV export can fetch HTML fragments by run id.
"""

import asyncio
import csv
import io
import json
import os
import time
import uuid
from collections import OrderedDict

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates
from markupsafe import Markup
from starlette.concurrency import run_in_threadpool

from . import views
from .icons import ICONS

TEMPLATES = Jinja2Templates(directory=os.path.join(views.WEB_DIR, "templates"))

SCENARIOS = [
    ("all", "All prepared flows"),
    ("all_correct", "All four correct"),
    ("gcd_only", "Only GCD models correct")
]

# Seconds spent on each inference stage in the simulation; 0 skips the stages.
SPEEDS = [("realtime", "Real time", 0.3), ("fast", "Fast", 0.07), ("instant", "Instant", 0.0)]

STAGES = ["Read flow", "Scale 21 features", "Fuzzy-GCD", "Build 5 × 5 maps", "Run 4 models"]

MAX_RUNS = 20


# Renders a lucide icon inline; size is in pixels and extra classes are optional.
def icon(name, size=16, cls=""):
    return Markup(
        f'<svg class="lucide {cls}" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
        f'aria-hidden="true">{ICONS[name]}</svg>'
    )


TEMPLATES.env.globals.update(icon=icon, MODELS=views.MODELS)
TEMPLATES.env.filters.update(
    pct=views.pct,
    num=views.format_number,
    feature_label=views.feature_label
)

RESULTS = views.load_results()
PROJECT = views.build_project(RESULTS)
HERO = views.hero_stream()
REFERENCE = {
    "label": f"Full test set ({RESULTS['test_set']['flows']:,} flows)",
    "accuracy": RESULTS["test_set"]["accuracy"]
}


# Creates the router. core is the backend.app module, which provides the model
# pipeline (classify_flow, read_flow_file, list_samples and friends).
def create_web_router(core):
    router = APIRouter(include_in_schema=False)
    runs = OrderedDict()

    # Stores a run's described flows and returns its id; old runs are evicted.
    def store_run(flows, kind, summary="", note=""):
        run_id = uuid.uuid4().hex[:12]
        runs[run_id] = {"flows": flows, "kind": kind, "summary": summary, "note": note}
        while len(runs) > MAX_RUNS:
            runs.popitem(last=False)
        return run_id

    # Looks up a stored run or answers 404 when it has expired.
    def get_run(run_id):
        if run_id not in runs:
            raise HTTPException(status_code=404, detail="These results have expired. Run the classification again.")
        return runs[run_id]

    # Renders a template to a string, for fragments sent inside JSON or SSE events.
    def render(name, **context):
        return TEMPLATES.get_template(name).render(**context)

    # Template context for the shared results view: insights, filters, table and
    # the first flow's breakdown.
    def results_context(run_id, run, active="all"):
        flows = run["flows"]
        scored = [f for f in flows if not f.get("error")]
        first = next((f for f in flows if not f.get("error")), None)
        unanimous = sum(f["votes"] == len(views.MODELS) for f in scored)

        return {
            "run_id": run_id,
            "title": "Simulation results" if run["kind"] == "simulation" else "Results",
            "summary": run["summary"],
            "note": run["note"],
            "agreement_text": f"All four models agree on {unanimous} of {len(scored)}." if scored else "",
            "errors": len(flows) - len(scored),
            "insights": views.build_insights(scored, REFERENCE if run["kind"] == "simulation" else None) if scored else None,
            **table_context(run_id, flows, active),
            "detail": views.build_flow_detail(first) if first else None
        }

    # Template context for the filter chips and flow table.
    def table_context(run_id, flows, active):
        return {
            "run_id": run_id,
            "filters": views.table_filters(flows, active),
            "active": active,
            "rows": [
                {"flow": f, "cells": None if f.get("error") else views.prediction_cells(f), "pips": None if f.get("error") else views.agreement_pips(f)}
                for f in flows
                if views.passes_filter(f, active)
            ],
            "selected": next((f["id"] for f in flows if not f.get("error")), None)
        }

    # ----------------------------------------------------------- pages

    # Dashboard: hero and the simulation over the prepared test flows.
    @router.get("/", response_class=HTMLResponse)
    def dashboard(request: Request):
        samples = core.list_samples()
        return TEMPLATES.TemplateResponse(request, "dashboard.html", {
            "page": "dashboard",
            "samples": samples,
            "directory": os.path.relpath(core.SAMPLES_DIR, core.BASE),
            "scenarios": [
                {"id": s, "label": label, "count": sum(1 for x in samples if s == "all" or x["group"] == s)}
                for s, label in SCENARIOS
            ],
            "speeds": SPEEDS,
            "stages": STAGES,
            "group_tags": views.GROUP_TAGS,
            "hero": HERO,
            "results": RESULTS,
            "best_accuracy": views.pct(max(RESULTS["test_set"]["accuracy"].values())),
            "class_count": len(core.classes),
            "scores": views.live_scores([])
        })

    # Old hash-style links (#/dashboard) and /dashboard both land on the dashboard.
    @router.get("/dashboard")
    def dashboard_alias():
        return RedirectResponse("/")

    # Upload page: queue files and classify them.
    @router.get("/classify", response_class=HTMLResponse)
    def classify_page(request: Request):
        return TEMPLATES.TemplateResponse(request, "classify.html", {"page": "classify"})

    # Project presentation built from the evaluation results.
    @router.get("/project", response_class=HTMLResponse)
    def project_page(request: Request):
        return TEMPLATES.TemplateResponse(request, "project.html", {"page": "project", "p": PROJECT})

    # ------------------------------------------------------- fragments

    # Classifies uploaded files and returns the results HTML plus a report per
    # file, which app.js shows in the queue.
    @router.post("/classify/run")
    async def classify_run(files: list[UploadFile] = File(...)):
        response = await core.predict_batch(files)
        flows = [views.describe_flow(f, i) for i, f in enumerate(response["flows"])]
        read = sum(1 for f in response["files"] if not f["error"])
        scored = sum(1 for f in flows if not f.get("error"))
        note = (
            f"Only the first {response['limit']} flows are classified per run; {response['skipped']} more were skipped."
            if response["skipped"] else ""
        )
        run_id = store_run(flows, "upload", f"{scored} flows classified from {read} {'file' if read == 1 else 'files'}.", note)

        return {
            "files": response["files"],
            "html": render("partials/results.html", **results_context(run_id, runs[run_id])) if flows else
            '<section class="results"><p class="results-empty">No flows were classified. Check the errors next to each file in the queue.</p></section>'
        }

    # Full results view for a stored run (used when a simulation finishes).
    @router.get("/runs/{run_id}/results", response_class=HTMLResponse)
    def run_results(request: Request, run_id: str):
        return TEMPLATES.TemplateResponse(request, "partials/results.html", results_context(run_id, get_run(run_id)))

    # Filter chips and flow table for a stored run, filtered by ?filter=.
    @router.get("/runs/{run_id}/table", response_class=HTMLResponse)
    def run_table(request: Request, run_id: str, filter: str = "all"):
        return TEMPLATES.TemplateResponse(request, "partials/flow_table.html", table_context(run_id, get_run(run_id)["flows"], filter))

    # Breakdown of one flow in a stored run.
    @router.get("/runs/{run_id}/flows/{index}", response_class=HTMLResponse)
    def run_flow(request: Request, run_id: str, index: int):
        flows = get_run(run_id)["flows"]
        if not 0 <= index < len(flows) or flows[index].get("error"):
            raise HTTPException(status_code=404, detail="No such flow")
        return TEMPLATES.TemplateResponse(request, "partials/flow_detail.html", {"detail": views.build_flow_detail(flows[index])})

    # Downloads a stored run's predictions as CSV.
    @router.get("/runs/{run_id}/csv")
    def run_csv(run_id: str):
        run = get_run(run_id)
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            ["file", "row", "flow_id", "true_label"]
            + [f"{m['name']} {field}" for m in views.MODELS for field in ("class", "confidence")]
            + ["models_agreeing"]
        )
        for f in run["flows"]:
            if f.get("error"):
                continue
            writer.writerow(
                [f["file_name"], f["row"] + 1, f["flow"]["flow_id"], f["flow"]["true_label"]]
                + [x for m in views.MODELS for x in (f["outputs"][m["name"]]["class"], f"{f['outputs'][m['name']]['confidence']:.4f}")]
                + [f["votes"]]
            )
        name = "flowlens-simulation.csv" if run["kind"] == "simulation" else "flowlens-results.csv"
        return Response(buffer.getvalue(), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{name}"'})

    # ------------------------------------------------------ simulation

    # Streams a simulation as server-sent events: a start event with the run id,
    # stage events while a flow is being classified, a flow event with the rendered
    # feed item, live scores and progress, and a final done event.
    @router.get("/simulate")
    async def simulate(request: Request, scenario: str = "all", speed: str = "realtime"):
        samples = [s for s in core.list_samples() if scenario == "all" or s["group"] == scenario]
        delay = next((d for key, _, d in SPEEDS if key == speed), SPEEDS[0][2])
        run_id = store_run([], "simulation", "", "Each scorecard also shows the model's accuracy on the full test set, for comparison with this sample.")
        run = runs[run_id]

        # One server-sent event with a JSON payload.
        def event(name, payload):
            return f"event: {name}\ndata: {json.dumps(payload)}\n\n"

        # Classifies one prepared file in a worker thread and describes its flows.
        def classify(sample, offset):
            df = pd.read_parquet(os.path.join(core.SAMPLES_DIR, sample["file_name"]))
            out = []
            for row in range(len(df)):
                entry = {"file_name": sample["file_name"], "row": row, "group": sample["group"], "test_position": sample["test_position"]}
                try:
                    entry.update(core.classify_flow(df.iloc[[row]].reset_index(drop=True)))
                except HTTPException as e:
                    entry["error"] = core.error_message(e)
                out.append(views.describe_flow(entry, offset + row))
            return out

        # Progress card HTML for the current state of the run.
        def progress(status, started, done):
            return render(
                "partials/sim_progress.html",
                done=done, total=len(samples), status=status,
                seconds=f"{time.monotonic() - started:.1f}" if status != "running" else None
            )

        async def stream():
            started = time.monotonic()
            yield event("start", {"run_id": run_id, "total": len(samples), "progress": progress("running", started, 0)})

            for index, sample in enumerate(samples):
                if await request.is_disconnected():
                    return

                task = asyncio.create_task(run_in_threadpool(classify, sample, len(run["flows"])))

                if delay:
                    for stage in range(len(STAGES)):
                        yield event("stage", {"now": render("partials/sim_now.html", sample=sample, stage=stage, stages=STAGES, group_tags=views.GROUP_TAGS)})
                        await asyncio.sleep(delay)

                try:
                    flows = await task
                except Exception as e:
                    yield event("failed", {"message": f"Couldn't classify {sample['file_name']}: {e}"})
                    return

                run["flows"].extend(flows)
                scored = [f for f in run["flows"] if not f.get("error")]
                run["summary"] = f"{len(scored)} prepared test flows classified."

                yield event("flow", {
                    "items": "".join(render("partials/feed_item.html", item=views.feed_item(f)) for f in flows),
                    "scores": render("partials/sim_scores.html", scores=views.live_scores(run["flows"])),
                    "progress": progress("running", started, index + 1)
                })

            yield event("done", {
                "run_id": run_id,
                "progress": progress("done", started, len(samples)),
                "now": render("partials/sim_now.html", sample=None, done=True)
            })

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )

    # Marks a simulation the user stopped, so its partial results read correctly.
    @router.post("/runs/{run_id}/stopped")
    def run_stopped(run_id: str):
        run = get_run(run_id)
        scored = sum(1 for f in run["flows"] if not f.get("error"))
        run["summary"] = f"{scored} prepared test flows classified before the run was stopped."
        return {"flows": len(run["flows"])}

    return router

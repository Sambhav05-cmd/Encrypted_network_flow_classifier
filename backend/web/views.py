"""View models for the web interface.

Everything the pages display is computed here and handed to the Jinja templates:
formatting, batch insights, the per-flow breakdown and the project slides.
"""

import json
import math
import os
import re
from collections import Counter

import numpy as np

WEB_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(WEB_DIR, "results.json")

MODELS = [
    {
        "name": "No-IBNN",
        "color": "var(--m-base)",
        "input": "Baseline input",
        "description": "Convolutional classifier on the 21 scaled flow features."
    },
    {
        "name": "IBNN",
        "color": "var(--m-ib)",
        "input": "Baseline input",
        "description": "The same encoder followed by a stochastic information bottleneck."
    },
    {
        "name": "Fuzzy-GCD",
        "color": "var(--m-gcd)",
        "input": "Fuzzy-GCD input",
        "description": "Four flow features swapped for fuzzy-GCD packet-size features."
    },
    {
        "name": "Fuzzy-GCD + IBNN",
        "color": "var(--m-gcdib)",
        "input": "Fuzzy-GCD input",
        "description": "Fuzzy-GCD input followed by the information bottleneck."
    }
]

GCD_FEATURES = ["k", "Mean residual", "Normalized residual", "Within tolerance"]
GCD_TOLERANCE = 4
HEAT_LIMIT = 3
PROTOCOLS = {6: "TCP", 17: "UDP", 1: "ICMP"}

GROUP_TAGS = {
    "all_correct": "All four correct",
    "gcd_only": "GCD models only",
    "other": "Prepared flow"
}

FEATURE_NAMES = [
    "outer_bytes", "outer_duration_ms", "outer_first_matched_time_ms", "outer_last_matched_time_ms",
    "outer_capture_duration_ms", "outer_packet_rate", "outer_byte_rate", "outer_bytes_in", "outer_bytes_out",
    "outer_min_piat_ms_in", "outer_mean_piat_ms_in", "outer_stddev_piat_ms_in", "outer_max_piat_ms_in",
    "outer_min_piat_ms_out", "outer_mean_piat_ms_out", "outer_stddev_piat_ms_out", "outer_max_piat_ms_out",
    "outer_min_piat_ms", "outer_mean_piat_ms", "outer_stddev_piat_ms", "outer_max_piat_ms"
]

PLAIN_NAMES = {
    "outer_bytes": "Total bytes",
    "outer_duration_ms": "Duration (ms)",
    "outer_first_matched_time_ms": "First matched time (ms)",
    "outer_last_matched_time_ms": "Last matched time (ms)",
    "outer_capture_duration_ms": "Capture duration (ms)",
    "outer_packet_rate": "Packet rate",
    "outer_byte_rate": "Byte rate",
    "outer_bytes_in": "Bytes inbound",
    "outer_bytes_out": "Bytes outbound"
}


# ---------------------------------------------------------------- formatting

# Formats a 0-1 fraction as a percentage string such as "83.8%".
def pct(value, digits=1):
    return f"{float(value) * 100:.{digits}f}%"


# Formats a number compactly: thousands separators for large values and four
# significant digits for small ones; non-finite values show as a dash.
def format_number(value):
    number = float(value)

    if not math.isfinite(number):
        return "–"
    if number == 0:
        return "0"
    if abs(number) >= 1000:
        return f"{number:,.0f}"

    return np.format_float_positional(number, precision=4, unique=False, fractional=False, trim="-")


# Formats a byte count with B, KB, MB, GB or TB units.
def format_bytes(value):
    scaled = float(value)
    units = ["B", "KB", "MB", "GB", "TB"]
    index = 0

    while abs(scaled) >= 1024 and index < len(units) - 1:
        scaled /= 1024
        index += 1

    return f"{scaled:,.1f} {units[index]}".replace(".0 ", " ") if index else f"{scaled:,.0f} B"


# Formats milliseconds as ms, s, min or h.
def format_duration(ms):
    value = float(ms)

    if not math.isfinite(value):
        return "–"
    if value < 1000:
        return f"{value:.0f} ms"
    if value < 60000:
        return f"{value / 1000:.1f} s"
    if value < 3600000:
        return f"{value / 60000:.1f} min"
    return f"{value / 3600000:.1f} h"


# Turns a feature column name such as outer_mean_piat_ms_in into a readable label.
def feature_label(name):
    if name in PLAIN_NAMES:
        return PLAIN_NAMES[name]

    match = re.match(r"^outer_(min|mean|stddev|max)_piat_ms(?:_(in|out))?$", name)
    if not match:
        return name

    stat = {"min": "Min", "mean": "Mean", "stddev": "Std. dev.", "max": "Max"}[match[1]]
    direction = {"in": ", inbound", "out": ", outbound"}.get(match[2], "")

    return f"{stat} inter-arrival{direction} (ms)"


# Arithmetic mean, or 0 for an empty list.
def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


# Builds the JSON payload app.js reads from a mark's data-tip attribute.
def tip(title, rows):
    return {"title": title, "rows": rows}


# ------------------------------------------------------------------ flows

# Adds an id, the majority class, its vote count and per-model correctness to a
# flow returned by classify_flow.
def describe_flow(flow, index):
    flow = {**flow, "id": index}

    if flow.get("error"):
        return flow

    label = flow["flow"]["true_label"]
    predictions = [flow["outputs"][model["name"]]["class"] for model in MODELS]
    majority, votes = Counter(predictions).most_common(1)[0]

    flow.update({
        "majority": majority,
        "votes": votes,
        "correct": [(name == label) if label else None for name in predictions],
        "display_name": flow["flow"]["flow_id"] or f"Row {flow['row'] + 1}",
        "source": f"{flow['file_name']}, row {flow['row'] + 1}"
    })

    return flow


# True when the flow came with a true application label.
def has_label(flow):
    return bool(flow.get("flow", {}).get("true_label"))


# A model output's most likely classes as (name, probability) pairs.
def top_predictions(output, count=3):
    return sorted(output["probabilities"].items(), key=lambda item: item[1], reverse=True)[:count]


# How decided a model is: the gap between its top two classes, and normalised
# entropy (0 is certain, 1 is a uniform guess).
def uncertainty(output):
    probabilities = sorted(output["probabilities"].values(), reverse=True)
    margin = probabilities[0] - (probabilities[1] if len(probabilities) > 1 else 0)
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    return margin, entropy / math.log(len(probabilities)) if len(probabilities) > 1 else 0.0


# Raw value of one named input feature for a flow.
def feature_value(flow, name):
    return next((f["raw"] for f in flow.get("features", []) if f["name"] == name), 0.0)


# Row cells for the flow table: each model's prediction, confidence and verdict.
def prediction_cells(flow):
    return [
        {
            "model": model,
            "class": flow["outputs"][model["name"]]["class"],
            "confidence": pct(flow["outputs"][model["name"]]["confidence"]),
            "correct": flow["correct"][index]
        }
        for index, model in enumerate(MODELS)
    ]


# Four agreement pips, coloured for the models that predict the majority class.
def agreement_pips(flow):
    return [
        model["color"] if flow["outputs"][model["name"]]["class"] == flow["majority"] else None
        for model in MODELS
    ]


FILTERS = [
    ("all", "All flows"),
    ("wrong", "Misclassified"),
    ("split", "Models disagree"),
    ("errors", "Errors")
]


# Whether a flow passes one of the table filters.
def passes_filter(flow, name):
    if name == "wrong":
        return not flow.get("error") and any(value is False for value in flow["correct"])
    if name == "split":
        return not flow.get("error") and flow["votes"] < len(MODELS)
    if name == "errors":
        return bool(flow.get("error"))
    return True


# Filter chips with counts for the flow table; hides filters that can't match.
def table_filters(flows, active):
    scored = [flow for flow in flows if not flow.get("error")]
    chips = []

    for name, label in FILTERS:
        if name == "wrong" and not any(has_label(flow) for flow in scored):
            continue
        if name == "errors" and len(scored) == len(flows):
            continue
        chips.append({
            "name": name,
            "label": label,
            "count": sum(passes_filter(flow, name) for flow in flows),
            "active": name == active
        })

    return chips


# ----------------------------------------------------------------- charts

# Rows for grouped horizontal bars: one bar per series with its width, text and a
# tooltip comparing all series. rows are (label, [value per series]).
def grouped_bars(rows, fmt, series=MODELS, maximum=None):
    top = maximum or max([1e-9] + [v for _, values in rows for v in values])

    return [
        {
            "label": label,
            "tip": tip(label, [
                {"color": s["color"], "label": s["name"], "value": fmt(v)}
                for s, v in zip(series, values)
            ]),
            "bars": [
                {
                    "color": s["color"],
                    "width": max(0.0, v) / top * 100,
                    "value": fmt(v),
                    "aria": f"{label}, {s['name']}: {fmt(v)}"
                }
                for s, v in zip(series, values)
            ]
        }
        for label, values in rows
    ]


# Rows for bars around a zero line: width is a share of half the track, and
# negative values extend left.
def diverging_bars(rows, fmt, series):
    limit = max([1e-9] + [abs(v) for _, values in rows for v in values])

    return {
        "ticks": [fmt(-limit), "0", fmt(limit)],
        "rows": [
            {
                "label": label,
                "tip": tip(label, [
                    {"color": s["color"], "label": s["name"], "value": fmt(v)}
                    for s, v in zip(series, values)
                ]),
                "bars": [
                    {
                        "color": s["color"],
                        "width": abs(v) / limit * 50,
                        "negative": v < 0,
                        "aria": f"{label}, {s['name']}: {fmt(v)}"
                    }
                    for s, v in zip(series, values)
                ]
            }
            for label, values in rows
        ]
    }


# ---------------------------------------------------------------- insights

# Batch-level insights for a list of classified flows: headline figures,
# scorecards and the analysis panels. reference adds full-test-set accuracy.
def build_insights(flows, reference=None):
    labelled = [flow for flow in flows if has_label(flow)]
    has_labels = bool(labelled)

    confidences = [flow["outputs"][m["name"]]["confidence"] for flow in flows for m in MODELS]
    unanimous = sum(flow["votes"] == len(MODELS) for flow in flows)
    majority_right = sum(flow["majority"] == flow["flow"]["true_label"] for flow in labelled)
    distinct = len({flow["outputs"][m["name"]]["class"] for flow in flows for m in MODELS})

    accuracies = [
        sum(bool(flow["correct"][index]) for flow in labelled) / len(labelled) if has_labels else None
        for index in range(len(MODELS))
    ]

    kpis = [
        {"icon": "boxes", "label": "Flows classified", "value": len(flows), "note": f"{distinct} different classes predicted"},
        {"icon": "handshake", "label": "All four agree", "value": pct(unanimous / len(flows), 0), "note": f"{unanimous} of {len(flows)} flows"},
        {"icon": "gauge", "label": "Mean confidence", "value": pct(mean(confidences), 0), "note": "Averaged over every model and flow"}
    ]

    if has_labels:
        best = max(range(len(MODELS)), key=lambda index: accuracies[index])
        kpis += [
            {"icon": "trophy", "label": "Best model", "value": MODELS[best]["name"], "note": f"{pct(accuracies[best])} accuracy", "tone": "accent"},
            {"icon": "crosshair", "label": "Majority vote accuracy", "value": pct(majority_right / len(labelled), 0), "note": f"{majority_right} of {len(labelled)} labelled flows"}
        ]
    else:
        kpis.append({"icon": "crosshair", "label": "True labels", "value": "None", "note": "Add an application_name column to score accuracy"})

    return {
        "kpis": kpis,
        "has_labels": has_labels,
        "scorecards": scorecards(flows, labelled, accuracies, reference),
        "classes": class_distribution(flows),
        "matrix": agreement_matrix(flows),
        "histograms": confidence_histograms(flows),
        "gcd": gcd_summary(flows),
        "effect": gcd_effect(labelled) if has_labels else None,
        "mixups": mixups(labelled) if has_labels else None
    }


# One card per model: accuracy (or mean confidence without labels), classes
# predicted, low-confidence count and confidence when right versus wrong.
def scorecards(flows, labelled, accuracies, reference):
    cards = []

    for index, model in enumerate(MODELS):
        outputs = [flow["outputs"][model["name"]] for flow in flows]
        mean_confidence = mean(o["confidence"] for o in outputs)
        accuracy = accuracies[index]
        right = [f["outputs"][model["name"]]["confidence"] for f in labelled if f["correct"][index]]
        wrong = [f["outputs"][model["name"]]["confidence"] for f in labelled if f["correct"][index] is False]
        headline = accuracy if accuracy is not None else mean_confidence

        cards.append({
            "model": model,
            "headline": pct(headline),
            "headline_label": "mean confidence" if accuracy is None else "accuracy",
            "meter": headline * 100,
            "reference": pct(reference["accuracy"][model["name"]]) if reference else None,
            "reference_label": reference["label"] if reference else None,
            "correct": f"{len(right)} / {len(labelled)}" if accuracy is not None else None,
            "mean_confidence": pct(mean_confidence, 0),
            "classes": len({o["class"] for o in outputs}),
            "unsure": sum(o["confidence"] < 0.5 for o in outputs),
            "right_wrong": f"{pct(mean(right), 0) if right else '–'} / {pct(mean(wrong), 0) if wrong else '–'}" if accuracy is not None else None
        })

    return cards


# Share of flows each model assigns to the eight most common classes, with the
# rest folded into "Other".
def class_distribution(flows, top_classes=8):
    counts = {}
    for flow in flows:
        for index, model in enumerate(MODELS):
            name = flow["outputs"][model["name"]]["class"]
            counts.setdefault(name, [0] * len(MODELS))[index] += 1

    ranked = sorted(counts.items(), key=lambda item: sum(item[1]), reverse=True)
    rows = [(label, [v / len(flows) for v in values]) for label, values in ranked[:top_classes]]

    rest = ranked[top_classes:]
    if rest:
        rows.append((
            f"Other ({len(rest)} classes)",
            [sum(values[i] for _, values in rest) / len(flows) for i in range(len(MODELS))]
        ))

    return grouped_bars(rows, lambda v: pct(v, 0))


# Pairwise matrix of how often two models predict the same class.
def agreement_matrix(flows):
    rows = []

    for a in MODELS:
        cells = []
        for b in MODELS:
            if a is b:
                cells.append({"self": True})
                continue
            share = sum(f["outputs"][a["name"]]["class"] == f["outputs"][b["name"]]["class"] for f in flows) / len(flows)
            cells.append({
                "self": False,
                "text": pct(share, 0),
                "strong": share > 0.6,
                "mix": round(12 + share * 78),
                "tip": tip(f"{a['name']} and {b['name']}", [{"label": "Same prediction", "value": pct(share, 0)}])
            })
        rows.append({"model": a, "cells": cells})

    return rows


# Histogram of each model's confidence across all flows, in 10% bins.
def confidence_histograms(flows, bins=10):
    out = []

    for model in MODELS:
        values = [flow["outputs"][model["name"]]["confidence"] for flow in flows]
        counts = [0] * bins
        for value in values:
            counts[min(bins - 1, int(value * bins))] += 1
        top = max([1] + counts)

        out.append({
            "model": model,
            "mean": pct(mean(values), 0),
            "bins": [
                {
                    "height": count / top * 100,
                    "aria": f"{model['name']}, {i * 10}–{i * 10 + 10}% confidence: {count} flows",
                    "tip": tip(f"{model['name']}, {i * 10}–{i * 10 + 10}% sure", [{"color": model["color"], "label": "Flows", "value": count}])
                }
                for i, count in enumerate(counts)
            ]
        })

    return out


# Fuzzy-GCD across the batch: share of packets on the lattice, mean residual and
# the six most common values of k.
def gcd_summary(flows):
    counts = Counter(flow["gcd"]["raw"]["best_k"] for flow in flows).most_common(6)
    most = max([1] + [count for _, count in counts])

    return {
        "within": pct(mean(f["gcd"]["raw"]["within_tolerance_fraction"] for f in flows), 0),
        "residual": f"{mean(f['gcd']['raw']['mean_residual'] for f in flows):.2f} B",
        "top_k": [
            {
                "k": format_number(k),
                "count": count,
                "width": count / most * 100,
                "tip": tip(f"k = {format_number(k)} bytes", [{"label": "Flows", "value": f"{count} ({pct(count / len(flows), 0)})"}])
            }
            for k, count in counts
        ]
    }


# Flows each GCD model gets right where its baseline is wrong (fixed) and the
# reverse (broken), plus flows only both GCD models get right.
def gcd_effect(flows):
    base, ib, gcd, gcd_ib = 0, 1, 2, 3

    def compare(with_gcd, without):
        return {
            "fixes": sum(bool(f["correct"][with_gcd]) and not f["correct"][without] for f in flows),
            "breaks": sum(not f["correct"][with_gcd] and bool(f["correct"][without]) for f in flows)
        }

    return {
        "pairs": [
            {"label": "Fuzzy-GCD vs No-IBNN", **compare(gcd, base)},
            {"label": "Fuzzy-GCD + IBNN vs IBNN", **compare(gcd_ib, ib)}
        ],
        "both": sum(
            bool(f["correct"][gcd]) and bool(f["correct"][gcd_ib]) and not f["correct"][base] and not f["correct"][ib]
            for f in flows
        )
    }


# Most frequent (true label, wrong majority class) pairs.
def mixups(flows):
    counts = Counter(
        (f["flow"]["true_label"], f["majority"])
        for f in flows
        if f["majority"] != f["flow"]["true_label"]
    )
    return [{"true": t, "predicted": p, "count": c} for (t, p), c in counts.most_common(6)]


# ------------------------------------------------------------- flow detail

# Rounds a value up to a readable axis maximum (1, 1.5, 2, 2.5, 3, 4, 5, 6 or 8 x 10^n).
def nice_max(value):
    if value <= 0:
        return 100
    magnitude = 10 ** math.floor(math.log10(value))
    return next(step * magnitude for step in [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10] if step * magnitude >= value)


# Geometry for the packet-size chart: one bar per packet, coloured by whether it
# lies within the tolerance of a multiple of k, plus lattice lines and ticks.
def packet_rhythm(sizes, k):
    if not sizes:
        return None

    width, height = 760, 220
    left, right, top_pad, bottom = 48, 12, 12, 26
    plot_w = width - left - right
    plot_h = height - top_pad - bottom
    top = nice_max(max(sizes))
    step = plot_w / len(sizes)
    bar_w = max(1.0, step * 0.72)

    def y(value):
        return top_pad + plot_h - value / top * plot_h

    multiples = int(top // k) if k > 0 else 0
    draw_lattice = 0 < multiples <= 48
    on_lattice = [abs(s - max(1, round(s / k)) * k) <= GCD_TOLERANCE for s in sizes]

    return {
        "width": width,
        "height": height,
        "left": left,
        "right_x": width - right,
        "baseline": top_pad + plot_h,
        "ticks": [{"y": y(t), "label": round(t)} for t in [0, top / 2, top]],
        "lattice": [y((i + 1) * k) for i in range(multiples)] if draw_lattice else [],
        "draw_lattice": draw_lattice,
        "bars": [
            {
                "x": left + i * step + (step - bar_w) / 2,
                "y": y(s),
                "w": bar_w,
                "h": max(1.0, top_pad + plot_h - y(s)),
                "rx": min(2.0, bar_w / 3),
                "on": on,
                "title": f"Packet {i + 1}: {format_number(s)} bytes"
            }
            for i, (s, on) in enumerate(zip(sizes, on_lattice))
        ],
        "count": len(sizes),
        "within": sum(on_lattice),
        "largest": format_number(max(sizes)),
        "k": format_number(k)
    }


# One 5x5 model input as heatmap cells: colour diverges around zero, padding
# cells are hatched and swapped GCD cells outlined.
def feature_map(matrix, replaced=()):
    cells = []

    for index, value in enumerate(v for row in matrix for v in row):
        padding = index >= 21
        strength = round(min(1.0, abs(value) / HEAT_LIMIT) * 85)
        tone = "var(--heat-pos)" if value >= 0 else "var(--heat-neg)"
        cells.append({
            "padding": padding,
            "swapped": index in replaced,
            "strong": strength > 45,
            "style": None if padding else f"background: color-mix(in srgb, {tone} {strength}%, var(--panel-solid))",
            "title": "Padding" if padding else f"Cell {index}: {value:.4f}",
            "text": "" if padding else f"{value:.2f}"
        })

    return cells


# Everything the flow breakdown shows: summary, pipeline, model votes, probability
# comparison, traffic profile, packet chart, GCD features and model inputs.
def build_flow_detail(flow):
    label = flow["flow"]["true_label"]
    raw = flow["gcd"]["raw"]
    # The sequence is padded with 0 and -1 for flows shorter than 255 packets; like the
    # Fuzzy-GCD features, only real (positive) packet sizes count.
    sizes = [size for size in flow["packet_sizes"] if size > 0]
    majority_right = (flow["majority"] == label) if label else None

    summary = [
        {"icon": "handshake", "label": "Consensus", "value": flow["majority"], "note": f"{flow['votes']} of 4 models agree", "tone": "accent"},
        {"icon": "gauge", "label": "Mean confidence", "value": pct(mean(flow["outputs"][m["name"]]["confidence"] for m in MODELS), 0), "note": "Across the four models"},
        {
            "icon": "crosshair",
            "label": "Against true label",
            "value": "No label" if majority_right is None else ("Correct" if majority_right else "Wrong"),
            "note": f"{sum(bool(c) for c in flow['correct'])} of 4 models correct" if label else "Add application_name to check",
            "tone": None if majority_right is None else ("good" if majority_right else "bad")
        },
        {"icon": "package", "label": "Packets observed", "value": len(sizes), "note": f"k = {format_number(raw['best_k'])} bytes"}
    ]

    pipeline = [
        ("Flow file", f"{flow['file_name']}, row {flow['row'] + 1}"),
        ("StandardScaler", "21 features scaled"),
        ("Fuzzy-GCD", f"k = {format_number(raw['best_k'])} B, {pct(raw['within_tolerance_fraction'], 0)} on lattice"),
        ("5 × 5 maps", "Baseline and GCD inputs"),
        ("4 models", f"Consensus {flow['majority']}")
    ]

    votes = []
    for index, model in enumerate(MODELS):
        output = flow["outputs"][model["name"]]
        margin, entropy = uncertainty(output)
        votes.append({
            "model": model,
            "class": output["class"],
            "correct": flow["correct"][index],
            "odds": [{"name": n, "value": pct(p), "width": p * 100} for n, p in top_predictions(output)],
            "margin": pct(margin, 0),
            "entropy": pct(entropy, 0)
        })

    leading = []
    for model in MODELS:
        for name, _ in top_predictions(flow["outputs"][model["name"]]):
            if name not in leading:
                leading.append(name)
    probability_rows = sorted(
        ((name, [flow["outputs"][m["name"]]["probabilities"].get(name, 0.0) for m in MODELS]) for name in leading),
        key=lambda row: max(row[1]),
        reverse=True
    )

    bytes_in = feature_value(flow, "outer_bytes_in")
    bytes_out = feature_value(flow, "outer_bytes_out")
    total = bytes_in + bytes_out

    profile = [
        ("timer", "Duration", format_duration(feature_value(flow, "outer_duration_ms")), ""),
        ("package", "Total bytes", format_bytes(feature_value(flow, "outer_bytes")), ""),
        ("gauge", "Packet rate", format_number(feature_value(flow, "outer_packet_rate")), ""),
        ("workflow", "Byte rate", format_number(feature_value(flow, "outer_byte_rate")), ""),
        ("timer", "Mean inter-arrival", format_duration(feature_value(flow, "outer_mean_piat_ms")), ""),
        (
            "bar-chart3",
            "Packet size",
            f"{format_number(mean(sizes))} B" if sizes else "–",
            f"{format_number(min(sizes))}–{format_number(max(sizes))} B, {len(set(sizes))} distinct" if sizes else ""
        )
    ]

    replaced_by = {feature: position for position, feature in enumerate(flow["gcd"]["replacement_indices"])}
    features = [
        {
            "index": index,
            "name": feature["name"],
            "label": feature_label(feature["name"]),
            "raw": format_number(feature["raw"]),
            "scaled": format_number(feature["scaled"]),
            "swap": (
                f"Fuzzy-GCD input uses {GCD_FEATURES[replaced_by[index]].lower()} here: "
                f"{format_number(flow['gcd']['scaled'][replaced_by[index]])}"
            ) if index in replaced_by else None
        }
        for index, feature in enumerate(flow["features"])
    ]

    return {
        "flow": flow,
        "label": label,
        "summary": summary,
        "pipeline": pipeline,
        "votes": votes,
        "probabilities": grouped_bars(probability_rows, pct, maximum=1),
        "profile": profile,
        "direction": {
            "in": format_bytes(bytes_in),
            "out": format_bytes(bytes_out),
            "in_share": pct(bytes_in / total if total else 0, 0),
            "out_share": pct(bytes_out / total if total else 0, 0),
            "width": bytes_in / total * 100 if total else 50
        },
        "rhythm": packet_rhythm(sizes, raw["best_k"]),
        "gcd_stats": [
            ("k", f"{format_number(raw['best_k'])} bytes", "Unit the packet sizes cluster around"),
            ("Mean residual", f"{format_number(raw['mean_residual'])} bytes", "Average distance to the nearest multiple"),
            ("Normalized residual", format_number(raw["normalized_residual"]), "Mean residual divided by k"),
            ("Within tolerance", pct(raw["within_tolerance_fraction"]), f"Packets within {GCD_TOLERANCE} bytes of a multiple")
        ],
        "features": features,
        "maps": [
            {"title": "Baseline input", "subtitle": "No-IBNN and IBNN", "cells": feature_map(flow["maps"]["normal"]), "swapped": False},
            {"title": "Fuzzy-GCD input", "subtitle": "Fuzzy-GCD and Fuzzy-GCD + IBNN", "cells": feature_map(flow["maps"]["gcd"], flow["gcd"]["replacement_indices"]), "swapped": True}
        ],
        "heat_limit": HEAT_LIMIT,
        "tolerance": GCD_TOLERANCE
    }


# One classified flow in the simulation feed: label, group, connection details and
# each model's prediction.
def feed_item(flow):
    if flow.get("error"):
        return {"error": f"{flow['file_name']}: {flow['error']}"}

    context = flow.get("context", {})
    endpoint = f"{context['dst_ip']}:{context['dst_port']}" if context.get("dst_ip") else ""
    protocol = PROTOCOLS.get(context.get("protocol"), f"Protocol {context['protocol']}" if context.get("protocol") else "")
    packets = f"{context['bidirectional_packets']} packets" if context.get("bidirectional_packets") else ""

    return {
        "label": flow["flow"]["true_label"] or "Unlabelled flow",
        "group": flow.get("group", "other"),
        "group_tag": GROUP_TAGS.get(flow.get("group"), GROUP_TAGS["other"]),
        "meta": ", ".join(x for x in [context.get("requested_server_name"), endpoint, protocol, packets] if x),
        "cells": prediction_cells(flow)
    }


# Running accuracy of each model over the labelled flows classified so far.
def live_scores(flows):
    labelled = [f for f in flows if not f.get("error") and has_label(f)]
    scores = []

    for index, model in enumerate(MODELS):
        share = sum(bool(f["correct"][index]) for f in labelled) / len(labelled) if labelled else 0
        scores.append({"model": model, "width": share * 100, "text": pct(share, 0) if labelled else "–"})

    return scores


# ---------------------------------------------------------------- results

# Loads the evaluation summary written by export_results.py.
def load_results():
    with open(RESULTS_FILE) as f:
        return json.load(f)


# Deterministic pseudo-random packet sizes for the hero visual: mostly whole
# multiples of k, with about one in five off the lattice.
def hero_stream(k=22, count=40, step=13, base=204):
    seed = 7

    def random():
        nonlocal seed
        seed = (seed * 16807) % 2147483647
        return seed / 2147483647

    bars = []
    for _ in range(count):
        units = 1 + math.floor(random() * 8)
        off = random() < 0.2
        bars.append({"height": units * k + (6 + random() * 8 if off else 0), "off": off})

    set_width = count * step
    return {
        "k": k,
        "set_width": set_width,
        "lattice": [base - (i + 1) * k for i in range(8)],
        "bars": [
            {"x": offset + i * step, "y": base - bar["height"], "w": step - 4, "h": bar["height"], "off": bar["off"]}
            for offset in (0, set_width)
            for i, bar in enumerate(bars)
        ]
    }


# Numbers for the project slides, all derived from results.json.
def build_project(results):
    dataset = results["dataset"]
    training = results["training"]
    overall = results["overall"]
    test_set = results["test_set"]
    accuracy = test_set["accuracy"]
    per_class = results["per_class"]
    best = max(overall, key=lambda row: row["accuracy"])

    def points(a, b):
        return f"{(a - b) * 100:.1f} pp"

    top_classes = per_class[:10]
    rest = sum(row["support"] for row in per_class[10:])
    class_rows = [(row["label"], row["support"]) for row in top_classes] + [(f"Other {len(per_class) - 10} classes", rest)]
    largest = max(v for _, v in class_rows)

    weak = sum(row["accuracy"]["Fuzzy-GCD + IBNN"] < 0.1 for row in per_class)

    metric_names = [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"), ("F1", "f1")]

    heat_rows = []
    for row in per_class[:14]:
        values = [row["accuracy"][m["name"]] for m in MODELS]
        top = max(values)
        heat_rows.append({
            "label": row["label"],
            "support": f"{row['support']:,}",
            "cells": [
                {
                    "text": pct(v),
                    "mix": round(v * 85),
                    "strong": v > 0.55,
                    "top": v == top and v > 0,
                    "tip": tip(f"{row['label']}, {m['name']}", [
                        {"label": "Accuracy", "value": pct(v)},
                        {"label": "Test flows", "value": f"{row['support']:,}"}
                    ])
                }
                for m, v in zip(MODELS, values)
            ]
        })

    gain_series = [
        {"name": "Fuzzy-GCD vs No-IBNN", "color": "var(--m-gcd)"},
        {"name": "Fuzzy-GCD + IBNN vs IBNN", "color": "var(--m-gcdib)"}
    ]

    def signed(v):
        return f"{'+' if v >= 0 else '−'}{abs(v):.2f}"

    top_gain = results["gcd_improvement"][0]
    confusion = results["confusions"]["Fuzzy-GCD + IBNN"][0]
    flows = test_set["flows"]

    return {
        "dataset": dataset,
        "training": training,
        "best": best,
        "best_accuracy": pct(best["accuracy"]),
        "baseline_accuracy": pct(accuracy["No-IBNN"]),
        "gain": points(best["accuracy"], accuracy["No-IBNN"]),
        "test_flows": f"{flows:,}",
        "largest_two": pct((per_class[0]["support"] + per_class[1]["support"]) / flows, 0),
        "class_names": (per_class[0]["label"], per_class[1]["label"]),
        "class_bars": [
            {"label": label, "value": f"{value:,}", "width": value / largest * 100, "tip": tip(label, [{"label": "Flows", "value": f"{value:,}"}])}
            for label, value in class_rows
        ],
        "replaced": [
            {"index": index, "feature": feature_label(FEATURE_NAMES[index]), "gcd": ["k", "mean residual", "normalized residual", "share within tolerance"][position]}
            for position, index in enumerate(training["replaced_feature_indices"])
        ],
        "betas": ", ".join(str(b) for b in training["betas"]),
        "metrics_chart": grouped_bars(
            [(label, [row[key] for row in overall]) for label, key in metric_names],
            pct,
            maximum=1
        ),
        "metric_names": [label for label, _ in metric_names],
        "metrics_table": [
            {"model": MODELS[i], "values": [pct(row[key]) for _, key in metric_names], "best": row["model"] == best["model"]}
            for i, row in enumerate(overall)
        ],
        "gcd_gain": points(accuracy["Fuzzy-GCD"], accuracy["No-IBNN"]),
        "ib_gain": points(accuracy["IBNN"], accuracy["No-IBNN"]),
        "ib_on_gcd": points(accuracy["Fuzzy-GCD + IBNN"], accuracy["Fuzzy-GCD"]),
        "heat_rows": heat_rows,
        "weak": weak,
        "class_count": len(per_class),
        "gain_series": gain_series,
        "gains": diverging_bars(
            [(row["label"], [row["vs_no_ibnn"], row["vs_ibnn"]]) for row in results["gcd_improvement"]],
            signed,
            gain_series
        ),
        "top_gain": {"label": top_gain["label"], "vs_no_ibnn": f"{top_gain['vs_no_ibnn']:.2f}", "vs_ibnn": f"{top_gain['vs_ibnn']:.2f}"},
        "agreement": [
            ("All four models correct", f"{test_set['all_four_correct']:,}", f"{pct(test_set['all_four_correct'] / flows)} of test flows"),
            ("Fixed by Fuzzy-GCD", f"{test_set['gcd_only_correct']:,}", "both GCD models right, both baselines wrong"),
            ("Broken by Fuzzy-GCD", f"{test_set['baseline_only_correct']:,}", f"net gain of {test_set['gcd_only_correct'] - test_set['baseline_only_correct']:,} flows"),
            ("All four wrong", f"{test_set['all_four_wrong']:,}", f"{pct(test_set['all_four_wrong'] / flows)} of test flows")
        ],
        "confusions": {
            model: [{**c, "count": f"{c['count']:,}"} for c in results["confusions"][model][:6]]
            for model in ["No-IBNN", "Fuzzy-GCD + IBNN"]
        },
        "findings": [
            f"Fuzzy-GCD packet-size features give the largest gain: accuracy rises from {pct(accuracy['No-IBNN'])} to {pct(accuracy['Fuzzy-GCD'])} with no change to the network.",
            f"The information bottleneck helps the baseline (+{points(accuracy['IBNN'], accuracy['No-IBNN'])}) but adds little once GCD features are present (+{points(accuracy['Fuzzy-GCD + IBNN'], accuracy['Fuzzy-GCD'])}).",
            f"GCD features fix {test_set['gcd_only_correct']:,} test flows that both baselines miss, against {test_set['baseline_only_correct']:,} they break.",
            f"The biggest per-class gain is {top_gain['label']}, whose F1 improves by {top_gain['vs_ibnn']:.2f} over IBNN.",
            f"The most common remaining error is {confusion['true']} predicted as {confusion['predicted']} ({confusion['count']:,} flows), and {weak} of {len(per_class)} classes stay below 10% accuracy."
        ]
    }

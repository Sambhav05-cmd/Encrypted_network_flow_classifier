"""Summarise the notebook's evaluation outputs for the web interface.

Reads the CSVs that analyse.py and plot.py write to backend/selected_flows/
and writes backend/web/results.json, which the Project page and the
dashboard's test-set comparison use. Run from the repository root:

    python backend/export_results.py
"""

import json
import os

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLOWS_DIR = os.path.join(BASE, "backend", "selected_flows")
OUTPUT = os.path.join(BASE, "backend", "web", "results.json")

MODELS = ["No-IBNN", "IBNN", "Fuzzy-GCD", "Fuzzy-GCD + IBNN"]

# Figures reported in the outputs of backend/PythonNotebook.ipynb.
DATASET = {
    "raw_flows": 226454,
    "columns": 126,
    "flows_after_filter": 225180,
    "classes": 35,
    "min_class_count": 100,
    "train_flows": 157626,
    "test_flows": 67554,
    "features": 21,
    "grid": "5 × 5",
    "packet_sizes_per_flow": 255
}

TRAINING = {
    "epochs": 30,
    "optimizer": "AdamW",
    "learning_rate": 1e-3,
    "schedule": "OneCycleLR",
    "batch_size": 256,
    "dropout": 0.1,
    "l2": 1e-4,
    "z_dim": 128,
    "betas": [0.01, 0.005, 0.0001, 0.00005, 0.000001],
    "seed": 42,
    "parameters": {"IBNN": 583171, "No-IBNN": 550147},
    "replaced_feature_indices": [5, 2, 7, 1]
}


# Reads the evaluation CSVs, computes per-class accuracy, confusions and agreement
# counts, and writes the summary JSON used by the web interface.
def main():
    overall = pd.read_csv(os.path.join(FLOWS_DIR, "plots", "overall_metrics.csv"))
    label_f1 = pd.read_csv(
        os.path.join(FLOWS_DIR, "plots", "gcd_label_metrics.csv"),
        index_col=0
    )
    predictions = pd.read_csv(os.path.join(FLOWS_DIR, "all_test_predictions.csv"))

    correct = {model: predictions[model] == predictions["true_label"] for model in MODELS}

    # Per-class support and accuracy (recall) for every model.
    per_class = []
    for label, group in predictions.groupby("true_label"):
        per_class.append({
            "label": label,
            "support": int(len(group)),
            "accuracy": {
                model: float((group[model] == label).mean()) for model in MODELS
            },
            "f1": {
                model: float(label_f1.loc[label, f"{model.replace(' ', '')}_F1"])
                for model in MODELS
            } if label in label_f1.index else None
        })
    per_class.sort(key=lambda row: row["support"], reverse=True)

    # Most frequent mistakes of the baseline and the best model.
    confusions = {}
    for model in ["No-IBNN", "Fuzzy-GCD + IBNN"]:
        wrong = predictions[~correct[model]]
        pairs = (
            wrong.groupby(["true_label", model]).size()
            .sort_values(ascending=False)
            .head(8)
        )
        confusions[model] = [
            {"true": true, "predicted": predicted, "count": int(count)}
            for (true, predicted), count in pairs.items()
        ]

    gcd_right = correct["Fuzzy-GCD"] & correct["Fuzzy-GCD + IBNN"]
    base_wrong = ~correct["No-IBNN"] & ~correct["IBNN"]
    base_right = correct["No-IBNN"] & correct["IBNN"]

    results = {
        "dataset": DATASET,
        "training": TRAINING,
        "overall": [
            {
                "model": row["Model"],
                "accuracy": row["Accuracy"],
                "precision": row["Precision"],
                "recall": row["Recall"],
                "f1": row["F1"]
            }
            for _, row in overall.iterrows()
        ],
        "test_set": {
            "flows": int(len(predictions)),
            "all_four_correct": int(pd.concat(correct, axis=1).all(axis=1).sum()),
            "all_four_wrong": int((~pd.concat(correct, axis=1)).all(axis=1).sum()),
            "gcd_only_correct": int((gcd_right & base_wrong).sum()),
            "baseline_only_correct": int((base_right & ~correct["Fuzzy-GCD"] & ~correct["Fuzzy-GCD + IBNN"]).sum()),
            "accuracy": {model: float(correct[model].mean()) for model in MODELS}
        },
        "per_class": per_class,
        "gcd_improvement": [
            {
                "label": label,
                "vs_no_ibnn": float(row["GCD_Improvement_NoIBNN"]),
                "vs_ibnn": float(row["GCD_Improvement_IBNN"])
            }
            for label, row in label_f1.sort_values(
                "Highest_GCD_Improvement", ascending=False
            ).head(10).iterrows()
        ],
        "confusions": confusions
    }

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {os.path.relpath(OUTPUT, BASE)}")


if __name__ == "__main__":
    main()

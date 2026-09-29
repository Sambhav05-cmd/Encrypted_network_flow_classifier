import os
import sys
import ast
import json
import numpy as np
import pandas as pd
import torch
import joblib

from backend.app import models


BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models")


with open(os.path.join(MODEL_DIR, "config.json")) as f:
    config = json.load(f)

classes = config["classes"]


feature_names = [
    "outer_bytes",
    "outer_duration_ms",
    "outer_first_matched_time_ms",
    "outer_last_matched_time_ms",
    "outer_capture_duration_ms",
    "outer_packet_rate",
    "outer_byte_rate",
    "outer_bytes_in",
    "outer_bytes_out",
    "outer_min_piat_ms_in",
    "outer_mean_piat_ms_in",
    "outer_stddev_piat_ms_in",
    "outer_max_piat_ms_in",
    "outer_min_piat_ms_out",
    "outer_mean_piat_ms_out",
    "outer_stddev_piat_ms_out",
    "outer_max_piat_ms_out",
    "outer_min_piat_ms",
    "outer_mean_piat_ms",
    "outer_stddev_piat_ms",
    "outer_max_piat_ms",
]


replace_idx_path = os.path.join(
    MODEL_DIR,
    "replace_idx.npy"
)

if os.path.exists(replace_idx_path):
    replace_idx = np.load(
        replace_idx_path
    ).astype(np.int64)
else:
    replace_idx = np.array(
        [5, 2, 7, 1],
        dtype=np.int64
    )


feature_scaler = joblib.load(
    os.path.join(
        MODEL_DIR,
        "feature_scaler.pkl"
    )
)


gcd_scaler = joblib.load(
    os.path.join(
        MODEL_DIR,
        "gcd_scaler.pkl"
    )
)


# Finds the size unit k (8-256 bytes) that packet sizes best fit as whole multiples and
# returns [k, mean residual, residual / k, share of packets within the tolerance].
def fuzzy_gcd_features(
    sizes,
    k_min=8,
    k_max=256,
    tolerance=4
):
    sizes = np.asarray(
        sizes,
        dtype=np.float32
    )

    sizes = sizes[
        np.isfinite(sizes) &
        (sizes > 0)
    ]

    if len(sizes) == 0:
        return np.zeros(
            4,
            dtype=np.float32
        )

    best_k = 8.0
    best_score = float("inf")

    for k in range(
        k_min,
        k_max + 1
    ):
        mult = np.maximum(
            1.0,
            np.rint(sizes / k)
        )

        res = np.abs(
            sizes - mult * k
        )

        score = res.mean()

        if score < best_score:
            best_score = score
            best_k = float(k)

    mult = np.maximum(
        1.0,
        np.rint(sizes / best_k)
    )

    res = np.abs(
        sizes - mult * best_k
    )

    return np.array([
        best_k,
        res.mean(),
        np.mean(res / best_k),
        np.mean(res <= tolerance)
    ], dtype=np.float32)


# Turns an outer_splt_ps value (list, array or string) into a float array;
# unreadable or missing values give an empty array.
def parse_packet_sizes(value):
    if isinstance(
        value,
        (list, tuple, np.ndarray)
    ):
        return np.asarray(
            value,
            dtype=np.float32
        )

    if pd.isna(value):
        return np.array(
            [],
            dtype=np.float32
        )

    try:
        return np.asarray(
            ast.literal_eval(str(value)),
            dtype=np.float32
        )
    except Exception:
        return np.array(
            [],
            dtype=np.float32
        )


# Zero-pads the scaled features to 25 values and reshapes them into the
# 1 x 1 x 5 x 5 tensor the models expect.
def make_map(x):
    padded = np.zeros(
        25,
        dtype=np.float32
    )

    padded[:21] = x

    return padded.reshape(
        1,
        1,
        5,
        5
    )


# Runs one model and returns the predicted class index and its probability.
def predict(model, x):
    with torch.no_grad():
        output = model(x)

        if isinstance(output, tuple):
            output = output[0]

        probabilities = torch.softmax(
            output,
            dim=1
        )

        prediction = int(
            torch.argmax(
                probabilities,
                dim=1
            ).item()
        )

        confidence = float(
            probabilities[
                0,
                prediction
            ].item()
        )

    return prediction, confidence


if len(sys.argv) != 2:
    print("Usage:")
    print("  python3 predict_one_flow.py <flow_file.parquet>")
    print("  python3 predict_one_flow.py <flow_file.csv>")
    sys.exit(1)


input_file = sys.argv[1]


if not os.path.exists(input_file):
    raise FileNotFoundError(
        f"Input file not found: {input_file}"
    )


if input_file.endswith(".parquet"):
    df = pd.read_parquet(
        input_file
    )
elif input_file.endswith(".csv"):
    df = pd.read_csv(
        input_file
    )
else:
    raise ValueError(
        "Input must be .parquet or .csv"
    )


if len(df) != 1:
    raise ValueError(
        f"Expected exactly one flow, but file contains {len(df)} rows."
    )


flow = df.iloc[0]


missing = [
    x
    for x in feature_names
    if x not in df.columns
]

if missing:
    raise ValueError(
        "Missing required features:\n" +
        "\n".join(missing)
    )


if "outer_splt_ps" not in df.columns:
    raise ValueError(
        "outer_splt_ps is required for GCD models."
    )


X = df[
    feature_names
].copy()


X = X.replace(
    [np.inf, -np.inf],
    np.nan
).fillna(0)


X_scaled = feature_scaler.transform(
    X
)[0].astype(
    np.float32
)


normal_map = make_map(
    X_scaled
)


packet_sizes = parse_packet_sizes(
    flow["outer_splt_ps"]
)


gcd_features = fuzzy_gcd_features(
    packet_sizes
)


gcd_scaled = gcd_scaler.transform(
    gcd_features.reshape(
        1,
        -1
    )
)[0].astype(
    np.float32
)


gcd_input = X_scaled.copy()


gcd_input[
    replace_idx
] = gcd_scaled


gcd_map = make_map(
    gcd_input
)


inputs = {
    "No-IBNN": normal_map,
    "IBNN": normal_map,
    "Fuzzy-GCD": gcd_map,
    "Fuzzy-GCD + IBNN": gcd_map,
}


print()
print("=" * 70)
print("FLOW")
print("=" * 70)

print(
    f"File: {input_file}"
)

if "flow_id" in df.columns:
    print(
        f"Flow ID: {flow['flow_id']}"
    )

if "application_name" in df.columns:
    print(
        f"True label: {flow['application_name']}"
    )


print()
print("=" * 70)
print("GCD FEATURES")
print("=" * 70)

print(
    f"Replacement indices: {replace_idx.tolist()}"
)

print(
    f"Raw GCD features: {gcd_features.tolist()}"
)

print(
    f"Scaled GCD features: {gcd_scaled.tolist()}"
)


print()
print("=" * 70)
print("PREDICTIONS")
print("=" * 70)


results = {}


for name in [
    "No-IBNN",
    "IBNN",
    "Fuzzy-GCD",
    "Fuzzy-GCD + IBNN"
]:
    x = torch.tensor(
        inputs[name],
        dtype=torch.float32
    )

    prediction, confidence = predict(
        models[name],
        x
    )

    predicted_class = classes[
        prediction
    ]

    results[name] = {
        "class": predicted_class,
        "confidence": confidence
    }

    print(
        f"{name:20s}: "
        f"{predicted_class:25s} "
        f"{confidence * 100:8.3f}%"
    )


print()
print("=" * 70)
print("GCD INPUT MAP")
print("=" * 70)

print(
    gcd_map[0, 0]
)


print()

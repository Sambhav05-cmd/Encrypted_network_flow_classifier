import os
import ast
import json
import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from backend.app import models


BASE = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE,
    "models"
)

DATA_DIR = os.path.join(
    BASE,
    "VPN-nonVPN-Dataset",
    "Data"
)

OUTPUT_DIR = os.path.join(
    BASE,
    "selected_flows"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


SEED = 42
TEST_SIZE = 0.30
MIN_CLASS_COUNT = 100


with open(
    os.path.join(
        MODEL_DIR,
        "config.json"
    )
) as f:
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


replace_idx = np.array(
    [5, 2, 7, 1],
    dtype=np.int64
)


# Turns a packet-size sequence (array, list or string) into a float array,
# keeping only non-negative numeric values.
def parse_sequence(v):
    if isinstance(
        v,
        np.ndarray
    ):
        a = v.tolist()

    elif isinstance(
        v,
        list
    ):
        a = v

    elif isinstance(
        v,
        str
    ):
        try:
            a = ast.literal_eval(v)
        except Exception:
            a = v.strip("[]")
            a = [] if not a else a.split(",")

    else:
        return np.empty(
            0,
            dtype=np.float32
        )

    out = []

    for x in a:
        try:
            z = float(x)

            if z >= 0:
                out.append(z)

        except Exception:
            pass

    return np.asarray(
        out,
        dtype=np.float32
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


# Zero-pads each row of scaled features to 25 values and reshapes the batch into
# N x 1 x 5 x 5 feature maps.
def make_map(X):
    out = np.zeros(
        (len(X), 25),
        dtype=np.float32
    )

    out[:, :X.shape[1]] = X

    return out.reshape(
        len(X),
        1,
        5,
        5
    )


print("Loading session 1...")
session1 = pd.read_parquet(
    os.path.join(
        DATA_DIR,
        "session1",
        "session1_flows.parquet"
    )
)

print("Loading session 2...")
session2 = pd.read_parquet(
    os.path.join(
        DATA_DIR,
        "session2",
        "session2_flows.parquet"
    )
)


df = pd.concat(
    [
        session1,
        session2
    ],
    ignore_index=True
)


print(
    f"Combined dataset: {df.shape}"
)


counts = df[
    "application_name"
].value_counts()

valid_classes = counts[
    counts >= MIN_CLASS_COUNT
].index


df = df[
    df["application_name"].isin(
        valid_classes
    )
].reset_index(
    drop=True
)


print(
    f"After class filtering: {df.shape}"
)

print(
    f"Classes: {df['application_name'].nunique()}"
)


print()
print("Creating exact Kaggle train/test split...")


train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=df["application_name"]
)


print(
    f"Train: {len(train_df)}"
)

print(
    f"Test: {len(test_df)}"
)


X_test = feature_scaler.transform(
    test_df[
        feature_names
    ].astype(
        np.float32
    )
).astype(
    np.float32
)


X_test_map = make_map(
    X_test
)


print(
    f"X_test_map: {X_test_map.shape}"
)


print()
print("Computing GCD features...")


test_sequences = test_df[
    "outer_splt_ps"
].map(
    parse_sequence
).tolist()


gcd_test = np.vstack([
    fuzzy_gcd_features(s)
    for s in test_sequences
])


gcd_test_scaled = gcd_scaler.transform(
    gcd_test
).astype(
    np.float32
)


X_test_gcd = X_test.copy()

X_test_gcd[
    :,
    replace_idx
] = gcd_test_scaled


X_test_gcd_map = make_map(
    X_test_gcd
)


print(
    f"GCD features: {gcd_test.shape}"
)

print(
    f"GCD map: {X_test_gcd_map.shape}"
)


# Predicts class indices for a batch of maps with a No-IBNN-style model,
# in chunks of 256 flows.
def predict_no_ib(
    model,
    X
):
    predictions = []

    model.eval()

    batch_size = 256

    with torch.no_grad():

        for start in range(
            0,
            len(X),
            batch_size
        ):
            end = min(
                start + batch_size,
                len(X)
            )

            x = torch.tensor(
                X[start:end],
                dtype=torch.float32
            )

            logits = model(x)

            predictions.append(
                torch.argmax(
                    logits,
                    dim=1
                ).cpu().numpy()
            )

    return np.concatenate(
        predictions
    )


# Predicts class indices for a batch of maps with an IBNN model, which returns
# (logits, MI); only the logits are used.
def predict_ib(
    model,
    X
):
    predictions = []

    model.eval()

    batch_size = 256

    with torch.no_grad():

        for start in range(
            0,
            len(X),
            batch_size
        ):
            end = min(
                start + batch_size,
                len(X)
            )

            x = torch.tensor(
                X[start:end],
                dtype=torch.float32
            )

            logits, _ = model(
                x
            )

            predictions.append(
                torch.argmax(
                    logits,
                    dim=1
                ).cpu().numpy()
            )

    return np.concatenate(
        predictions
    )


print()
print("Running No-IBNN...")

pred_no_ibnn = predict_no_ib(
    models["No-IBNN"],
    X_test_map
)


print("Running IBNN...")

pred_ibnn = predict_ib(
    models["IBNN"],
    X_test_map
)


print("Running Fuzzy-GCD...")

pred_fuzzy_gcd = predict_no_ib(
    models["Fuzzy-GCD"],
    X_test_gcd_map
)


print("Running Fuzzy-GCD + IBNN...")

pred_fuzzy_gcd_ibnn = predict_ib(
    models["Fuzzy-GCD + IBNN"],
    X_test_gcd_map
)


y_test = np.array([
    classes.index(x)
    for x in test_df[
        "application_name"
    ]
])


true_labels = test_df[
    "application_name"
].to_numpy()


results = pd.DataFrame({
    "test_position": np.arange(
        len(test_df)
    ),

    "original_index": test_df.index.to_numpy(),

    "true_label": true_labels,

    "No-IBNN": [
        classes[x]
        for x in pred_no_ibnn
    ],

    "IBNN": [
        classes[x]
        for x in pred_ibnn
    ],

    "Fuzzy-GCD": [
        classes[x]
        for x in pred_fuzzy_gcd
    ],

    "Fuzzy-GCD + IBNN": [
        classes[x]
        for x in pred_fuzzy_gcd_ibnn
    ]
})


results["all_four_correct"] = (
    (pred_no_ibnn == y_test) &
    (pred_ibnn == y_test) &
    (pred_fuzzy_gcd == y_test) &
    (pred_fuzzy_gcd_ibnn == y_test)
)


results["gcd_only"] = (
    (pred_fuzzy_gcd == y_test) &
    (pred_fuzzy_gcd_ibnn == y_test) &
    (pred_no_ibnn != y_test) &
    (pred_ibnn != y_test)
)


# Picks up to count rows where condition_column is true, with at most one
# row per true label, so the examples cover different applications.
def select_distinct(
    table,
    condition_column,
    count=10
):
    selected = []
    used_labels = set()

    for _, row in table[
        table[condition_column]
    ].iterrows():

        label = row[
            "true_label"
        ]

        if label in used_labels:
            continue

        selected.append(
            row
        )

        used_labels.add(
            label
        )

        if len(selected) == count:
            break

    if not selected:
        return pd.DataFrame(
            columns=table.columns
        )

    return pd.DataFrame(
        selected
    )


selected_all = select_distinct(
    results,
    "all_four_correct",
    10
)


selected_gcd = select_distinct(
    results,
    "gcd_only",
    10
)


print()
print("=" * 90)
print("10 FLOWS: ALL FOUR MODELS CORRECT")
print("=" * 90)

print(
    selected_all[
        [
            "test_position",
            "original_index",
            "true_label",
            "No-IBNN",
            "IBNN",
            "Fuzzy-GCD",
            "Fuzzy-GCD + IBNN"
        ]
    ].to_string(
        index=False
    )
)


print()
print("=" * 90)
print("10 FLOWS: GCD MODELS CORRECT, OTHER TWO WRONG")
print("=" * 90)

print(
    selected_gcd[
        [
            "test_position",
            "original_index",
            "true_label",
            "No-IBNN",
            "IBNN",
            "Fuzzy-GCD",
            "Fuzzy-GCD + IBNN"
        ]
    ].to_string(
        index=False
    )
)


selected_all.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "all_four_correct_selected.csv"
    ),
    index=False
)


selected_gcd.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "gcd_only_selected.csv"
    ),
    index=False
)


results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "all_test_predictions.csv"
    ),
    index=False
)


# Writes each selected test flow to its own single-row .parquet file named
# <prefix>_<nn>_testpos_<position>_label_<label>.parquet.
def save_flows(
    selected,
    prefix
):
    for number, (_, row) in enumerate(
        selected.iterrows(),
        start=1
    ):
        test_position = int(
            row["test_position"]
        )

        original_index = int(
            row["original_index"]
        )

        flow = test_df.iloc[
            [test_position]
        ].copy()

        label = str(
            row["true_label"]
        ).replace(
            "/",
            "_"
        ).replace(
            " ",
            "_"
        )

        filename = (
            f"{prefix}_{number:02d}_"
            f"testpos_{test_position}_"
            f"label_{label}.parquet"
        )

        path = os.path.join(
            OUTPUT_DIR,
            filename
        )

        flow.to_parquet(
            path,
            index=False
        )

        print(
            f"Saved: {path}"
        )


print()
print("Saving all-correct flows...")

save_flows(
    selected_all,
    "all_correct"
)


print()
print("Saving GCD-only flows...")

save_flows(
    selected_gcd,
    "gcd_only"
)


print()
print("=" * 90)
print("DONE")
print("=" * 90)

print(
    f"All-four distinct-label examples: "
    f"{len(selected_all)}"
)

print(
    f"GCD-only distinct-label examples: "
    f"{len(selected_gcd)}"
)

print(
    f"Output directory: {OUTPUT_DIR}"
)

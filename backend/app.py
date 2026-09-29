import os
import sys
import ast
import json
import math
import io
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib

from typing import List

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE, "models")
DEVICE = torch.device("cpu")

app = FastAPI(title="Network Traffic Classifier")

app.add_middleware(
    CORSMiddleware,
    # The pages are served by this app itself; this lets local tools on any port
    # call the JSON API too.
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


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

num_classes = int(config["num_classes"])
grid_side = int(config["grid_side"])
z_dim = int(config["z_dim"])
dropout = float(config["dropout"])


# Convolutional feature extractor from the IBNN paper: 5x5, 5x5, 3x3, 3x3 convolutions.
# Returns the last two layers' feature maps concatenated along the channel axis.
class PaperConvEncoder(nn.Module):
    # Four convolution layers with 32, 32, 64 and 64 channels.
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            1,
            32,
            5,
            padding=2
        )

        self.conv2 = nn.Conv2d(
            32,
            32,
            5,
            padding=2
        )

        self.conv3 = nn.Conv2d(
            32,
            64,
            3,
            padding=1
        )

        self.conv4 = nn.Conv2d(
            64,
            64,
            3,
            padding=1
        )

    # Applies the convolutions with ReLU and concatenates the outputs of the
    # two 3x3 layers.
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))
        x4 = F.relu(self.conv4(x3))

        return torch.cat(
            [x3, x4],
            dim=1
        )


# Stochastic information-bottleneck layer: maps features to a Gaussian latent z
# and estimates the mutual information I(X; Z) used in the IBNN loss.
class IBLayer(nn.Module):
    # Linear heads for the latent mean and log-variance.
    def __init__(self, in_dim, z_dim):
        super().__init__()

        self.mu = nn.Linear(
            in_dim,
            z_dim
        )

        self.logvar = nn.Linear(
            in_dim,
            z_dim
        )

    # Samples z while training (uses the mean at inference) and estimates I(X; Z)
    # from pairwise KL divergences between the batch's Gaussians.
    def forward(self, x, num_samples=1):
        mu = self.mu(x)

        logvar = torch.clamp(
            self.logvar(x),
            -10.0,
            10.0
        )

        var = torch.exp(logvar)
        std = torch.exp(0.5 * logvar)

        z = mu

        if self.training or num_samples > 1:
            if num_samples == 1:
                z = mu + std * torch.randn_like(std)
            else:
                eps = torch.randn(
                    num_samples,
                    *mu.shape,
                    device=mu.device
                )

                z = (
                    mu.unsqueeze(0)
                    + eps * std.unsqueeze(0)
                ).mean(0)

        mu_i = mu.unsqueeze(1)
        mu_j = mu.unsqueeze(0)

        var_i = var.unsqueeze(1)
        var_j = var.unsqueeze(0)

        logvar_i = logvar.unsqueeze(1)
        logvar_j = logvar.unsqueeze(0)

        kl = 0.5 * (
            logvar_j
            - logvar_i
            + (
                var_i
                + (mu_i - mu_j).pow(2)
            ) / var_j
            - 1.0
        )

        kl = kl.sum(-1)

        b = x.size(0)

        if b == 1:
            mi = torch.zeros(
                (),
                device=x.device
            )
        else:
            mi = -(
                torch.logsumexp(
                    -kl,
                    dim=1
                )
                - math.log(b)
            ).mean()

        return z, mi


# Classifier with the information bottleneck: encoder, 128-unit layer, IB layer,
# then 128- and 256-unit layers and the output. Returns logits and the MI estimate.
class IBNN(nn.Module):
    # Encoder, 128-unit extractor, IB layer and the 128/256-unit classifier head.
    def __init__(
        self,
        num_classes,
        z_dim=128,
        dropout=0.1
    ):
        super().__init__()

        self.encoder = PaperConvEncoder()

        with torch.no_grad():
            d = self.encoder(
                torch.zeros(
                    1,
                    1,
                    grid_side,
                    grid_side
                )
            ).flatten(1).shape[1]

        self.extract = nn.Sequential(
            nn.Linear(d, 128),
            nn.ReLU(True),
            nn.Dropout(dropout)
        )

        self.ib = IBLayer(
            128,
            z_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(
                256,
                num_classes
            )
        )

    # Returns class logits and the mutual-information estimate for the batch.
    def forward(
        self,
        x,
        num_samples=1
    ):
        x = self.encoder(x).flatten(1)
        x = self.extract(x)

        z, mi = self.ib(
            x,
            num_samples
        )

        return self.classifier(z), mi


# Same network as IBNN with the bottleneck layer removed, used by the No-IBNN
# and Fuzzy-GCD models.
class NoIBNN(nn.Module):
    # Encoder, 128-unit extractor and the 128/256-unit classifier head.
    def __init__(
        self,
        num_classes,
        dropout=0.1
    ):
        super().__init__()

        self.encoder = PaperConvEncoder()

        with torch.no_grad():
            d = self.encoder(
                torch.zeros(
                    1,
                    1,
                    grid_side,
                    grid_side
                )
            ).flatten(1).shape[1]

        self.extract = nn.Sequential(
            nn.Linear(d, 128),
            nn.ReLU(True),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(
                256,
                num_classes
            )
        )

    # Returns class logits.
    def forward(self, x):
        x = self.encoder(x).flatten(1)
        x = self.extract(x)

        return self.classifier(x)


# Builds the IBNN or No-IBNN architecture, loads trained weights from path and
# switches the model to evaluation mode on the CPU.
def load_model(path, ib=False):
    if ib:
        model = IBNN(
            num_classes,
            z_dim,
            dropout
        )
    else:
        model = NoIBNN(
            num_classes,
            dropout
        )

    state = torch.load(
        path,
        map_location=DEVICE
    )

    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    return model


models = {
    "No-IBNN": load_model(
        os.path.join(
            MODEL_DIR,
            "no_ibnn.pt"
        )
    ),

    "IBNN": load_model(
        os.path.join(
            MODEL_DIR,
            "ibnn.pt"
        ),
        True
    ),

    "Fuzzy-GCD": load_model(
        os.path.join(
            MODEL_DIR,
            "fuzzy_gcd.pt"
        )
    ),

    "Fuzzy-GCD + IBNN": load_model(
        os.path.join(
            MODEL_DIR,
            "fuzzy_gcd_ibnn.pt"
        ),
        True
    )
}


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


# Turns an outer_splt_ps value (list, array or string like "[60, 1500]") into a
# float array; unreadable or missing values give an empty array.
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


# Zero-pads the 21 scaled features to 25 values and reshapes them into the
# 1 x 1 x 5 x 5 tensor layout the convolutional models expect.
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


# Validates a one-row DataFrame and builds both model inputs: the baseline 5x5 map and
# the Fuzzy-GCD map with four cells replaced by scaled GCD features.
def prepare_flow(df):
    missing = [
        x
        for x in feature_names
        if x not in df.columns
    ]

    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Missing required features",
                "missing_features": missing
            }
        )

    if "outer_splt_ps" not in df.columns:
        raise HTTPException(
            status_code=422,
            detail={
                "message":
                "outer_splt_ps is required for GCD models."
            }
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

    flow = df.iloc[0]

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

    return (
        flow,
        normal_map,
        gcd_map,
        gcd_features,
        gcd_scaled
    )


# Runs one model on a prepared map and returns the predicted class index,
# its probability and the full softmax distribution.
def predict(model, x):
    with torch.no_grad():
        tensor = torch.tensor(
            x,
            dtype=torch.float32
        )

        output = model(
            tensor
        )

        if isinstance(
            output,
            tuple
        ):
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

    return prediction, confidence, probabilities


# Runs the named model and formats its output as class name, index, confidence
# and the probability of every class.
def build_prediction(
    model_name,
    x
):
    prediction, confidence, probabilities = predict(
        models[model_name],
        x
    )

    return {
        "class": classes[prediction],
        "class_index": prediction,
        "confidence": confidence,
        "probabilities": {
            classes[i]: float(
                probabilities[
                    0,
                    i
                ].item()
            )
            for i in range(
                len(classes)
            )
        }
    }


# Request body for /predict: a single flow as a column-name to value mapping.
class FlowRequest(BaseModel):
    flow: dict


# Reports that the server is up, with the class list and feature count.
@app.get("/health")
def health():
    return {
        "status": "ok",
        "classes": classes,
        "feature_count": len(feature_names)
    }


# Lists the 21 feature columns every uploaded flow must contain.
@app.get("/features")
def features():
    return {
        "feature_names": feature_names,
        "feature_count": len(feature_names)
    }


# Classifies one flow sent as JSON and returns the four model outputs and
# its Fuzzy-GCD features.
@app.post("/predict")
def predict_json(
    request: FlowRequest
):
    flow = request.flow

    df = pd.DataFrame(
        [flow]
    )

    (
        _,
        normal_map,
        gcd_map,
        gcd_features,
        gcd_scaled
    ) = prepare_flow(df)

    return {
        "outputs": run_models(
            normal_map,
            gcd_map
        ),

        "fuzzy_gcd": {
            "best_k": float(
                gcd_features[0]
            ),
            "mean_residual": float(
                gcd_features[1]
            ),
            "normalized_residual": float(
                gcd_features[2]
            ),
            "within_tolerance_fraction": float(
                gcd_features[3]
            )
        }
    }


MAX_BATCH_FLOWS = 500
MAX_PACKET_SIZES = 512

# Prepared single-flow files used by the dashboard simulation.
SAMPLES_DIR = os.environ.get(
    "FLOWLENS_SAMPLES_DIR",
    os.path.join(BASE, "backend", "selected_flows")
)

# Written by analyse.py: <group>_<nn>_testpos_<position>_label_<label>.parquet
SAMPLE_NAME = re.compile(
    r"^(?P<group>all_correct|gcd_only)_(?P<number>\d+)"
    r"_testpos_(?P<position>\d+)_label_(?P<label>.+)\.parquet$"
)

SAMPLE_GROUPS = {
    "all_correct": "All four models classify these test flows correctly",
    "gcd_only": "Only the two Fuzzy-GCD models classify these test flows correctly",
    "other": "Other prepared flows"
}

# Flow metadata shown for context. None of it is a model input.
CONTEXT_COLUMNS = [
    "src_ip",
    "src_port",
    "dst_ip",
    "dst_port",
    "protocol",
    "requested_server_name",
    "application_category_name",
    "bidirectional_packets",
    "bidirectional_bytes",
    "bidirectional_duration_ms"
]


# Runs all four models: the baseline map feeds No-IBNN and IBNN, the GCD map
# feeds Fuzzy-GCD and Fuzzy-GCD + IBNN.
def run_models(normal_map, gcd_map):
    return {
        "No-IBNN": build_prediction(
            "No-IBNN",
            normal_map
        ),

        "IBNN": build_prediction(
            "IBNN",
            normal_map
        ),

        "Fuzzy-GCD": build_prediction(
            "Fuzzy-GCD",
            gcd_map
        ),

        "Fuzzy-GCD + IBNN": build_prediction(
            "Fuzzy-GCD + IBNN",
            gcd_map
        )
    }


# Returns a column's value as text, or an empty string when the column is
# missing or empty.
def optional_text(flow, column):
    if column not in flow.index or pd.isna(flow[column]):
        return ""

    return str(flow[column])


# Collects connection details (endpoints, protocol, server name, packet count)
# for display only; none of them are model inputs.
def flow_context(flow):
    context = {}

    for column in CONTEXT_COLUMNS:
        if column not in flow.index or pd.isna(flow[column]):
            continue

        value = flow[column]

        if isinstance(value, (np.integer, np.floating)):
            value = value.item()

        context[column] = value if isinstance(value, (int, float)) else str(value)

    return context


# Runs the full pipeline on a one-row DataFrame and describes every stage:
# labels, raw and scaled features, packet sizes, GCD features, predictions and maps.
def classify_flow(df):
    (
        flow,
        normal_map,
        gcd_map,
        gcd_features,
        gcd_scaled
    ) = prepare_flow(df)

    raw = df[feature_names].replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0).iloc[0]

    packet_sizes = parse_packet_sizes(
        flow["outer_splt_ps"]
    )

    return {
        "flow": {
            "flow_id": optional_text(flow, "flow_id"),
            "true_label": optional_text(flow, "application_name")
        },

        "features": [
            {
                "name": name,
                "raw": float(raw[name]),
                "scaled": float(normal_map.flat[i])
            }
            for i, name in enumerate(feature_names)
        ],

        "context": flow_context(flow),

        "packet_sizes": [
            float(x)
            for x in packet_sizes[:MAX_PACKET_SIZES]
        ],

        "gcd": {
            "raw": {
                "best_k": float(
                    gcd_features[0]
                ),
                "mean_residual": float(
                    gcd_features[1]
                ),
                "normalized_residual": float(
                    gcd_features[2]
                ),
                "within_tolerance_fraction": float(
                    gcd_features[3]
                )
            },

            "scaled": [
                float(x)
                for x in gcd_scaled
            ],

            "replacement_indices": [
                int(x)
                for x in replace_idx
            ]
        },

        "outputs": run_models(
            normal_map,
            gcd_map
        ),

        "maps": {
            "normal": normal_map[
                0,
                0
            ].tolist(),

            "gcd": gcd_map[
                0,
                0
            ].tolist()
        }
    }


# Reads an uploaded .parquet or .csv file into a DataFrame, rejecting other
# file types and unreadable files with a clear HTTP error.
async def read_flow_file(file):
    name = (file.filename or "").lower()

    if not name:
        raise HTTPException(
            status_code=400,
            detail="No file selected"
        )

    if not name.endswith((".parquet", ".csv")):
        raise HTTPException(
            status_code=400,
            detail="Only .parquet and .csv files are supported"
        )

    contents = await file.read()

    try:
        if name.endswith(".parquet"):
            return pd.read_parquet(
                io.BytesIO(contents)
            )

        return pd.read_csv(
            io.BytesIO(contents)
        )

    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Could not read {file.filename}: {str(e)}"
        )


# Turns an HTTPException's detail into one readable sentence, including the
# names of any missing columns.
def error_message(error):
    detail = error.detail

    if isinstance(detail, dict):
        message = detail.get("message", "Prediction failed")

        if detail.get("missing_features"):
            message += ": " + ", ".join(
                detail["missing_features"]
            )

        return message

    return str(detail)


# Classifies an uploaded file that must contain exactly one flow.
@app.post("/predict-file")
async def predict_file(
    file: UploadFile = File(...)
):
    df = await read_flow_file(file)

    if len(df) != 1:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Expected exactly one flow",
                "rows_found": len(df)
            }
        )

    return {
        "file_name": file.filename,
        **classify_flow(df)
    }


# Classifies every row of every uploaded file, up to MAX_BATCH_FLOWS flows, and
# reports per file how many rows were read or why the file failed.
@app.post("/predict-batch")
async def predict_batch(
    files: List[UploadFile] = File(...)
):
    file_reports = []
    flows = []
    skipped = 0

    for file in files:
        try:
            df = await read_flow_file(file)
        except HTTPException as e:
            file_reports.append({
                "file_name": file.filename,
                "rows": 0,
                "error": error_message(e)
            })
            continue

        missing = [
            x
            for x in feature_names + ["outer_splt_ps"]
            if x not in df.columns
        ]

        file_reports.append({
            "file_name": file.filename,
            "rows": len(df),
            "error": (
                "Missing required columns: " + ", ".join(missing)
                if missing
                else None
            )
        })

        if missing:
            continue

        for row in range(len(df)):
            if len(flows) >= MAX_BATCH_FLOWS:
                skipped += 1
                continue

            entry = {
                "file_name": file.filename,
                "row": row
            }

            try:
                entry.update(
                    classify_flow(
                        df.iloc[[row]].reset_index(drop=True)
                    )
                )
            except HTTPException as e:
                entry["error"] = error_message(e)
            except Exception as e:
                entry["error"] = f"Prediction failed: {str(e)}"

            flows.append(entry)

    return {
        "files": file_reports,
        "flows": flows,
        "skipped": skipped,
        "limit": MAX_BATCH_FLOWS
    }


# Lists the prepared .parquet files in SAMPLES_DIR, with the group, test position
# and label parsed from each file name.
def list_samples():
    if not os.path.isdir(SAMPLES_DIR):
        return []

    samples = []

    for name in sorted(os.listdir(SAMPLES_DIR)):
        if not name.endswith(".parquet"):
            continue

        match = SAMPLE_NAME.match(name)

        samples.append({
            "file_name": name,
            "group": match["group"] if match else "other",
            "test_position": int(match["position"]) if match else None,
            "label": match["label"] if match else "",
            "size": os.path.getsize(os.path.join(SAMPLES_DIR, name))
        })

    return samples


# Returns the prepared flow files the dashboard simulation can run.
@app.get("/samples")
def samples():
    return {
        "directory": os.path.relpath(SAMPLES_DIR, BASE),
        "groups": SAMPLE_GROUPS,
        "samples": list_samples()
    }


# Classifies one prepared flow file by name. Only names returned by list_samples
# are accepted, so requests can't read other files.
@app.post("/samples/{file_name}/classify")
def classify_sample(file_name: str):
    sample = next(
        (x for x in list_samples() if x["file_name"] == file_name),
        None
    )

    if sample is None:
        raise HTTPException(
            status_code=404,
            detail=f"No prepared flow named {file_name}"
        )

    df = pd.read_parquet(
        os.path.join(SAMPLES_DIR, sample["file_name"])
    )

    flows = []

    for row in range(min(len(df), MAX_BATCH_FLOWS)):
        entry = {
            "file_name": file_name,
            "row": row,
            "group": sample["group"],
            "test_position": sample["test_position"]
        }

        try:
            entry.update(
                classify_flow(
                    df.iloc[[row]].reset_index(drop=True)
                )
            )
        except HTTPException as e:
            entry["error"] = error_message(e)

        flows.append(entry)

    return {
        **sample,
        "flows": flows
    }


# The web interface: server-rendered pages, HTMX fragments and static files.
from fastapi.staticfiles import StaticFiles  # noqa: E402

from backend.web.routes import create_web_router  # noqa: E402

app.mount(
    "/static",
    StaticFiles(directory=os.path.join(BASE, "backend", "web", "static")),
    name="static"
)
app.include_router(create_web_router(sys.modules[__name__]))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )

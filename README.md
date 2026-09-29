# Encrypted Network Flow Classifier

A machine learning-based system for classifying encrypted network traffic flows into application-level categories using statistical flow features.

The project evaluates four classification models with and without **Fuzzy-GCD features** and **Information Bottleneck Neural Network (IBNN)** processing:

* No-IBNN
* IBNN
* Fuzzy-GCD
* Fuzzy-GCD + IBNN

Everything runs in Python: one FastAPI server loads the models, serves the JSON API, and renders the web interface (FlowLens) with Jinja2 templates and HTMX. There is no Node.js, npm or build step.

## Project Overview

Encrypted traffic cannot be classified using application-layer payload contents because the payload is protected by encryption. This project instead uses statistical characteristics of network flows, such as:

* Flow duration
* Packet rate
* Byte rate
* Bytes in/out
* Packet inter-arrival times
* Minimum, mean, standard deviation and maximum packet inter-arrival times

The feature pipeline uses 21 flow-level features.

Fuzzy-GCD processing extracts additional information from packet-size sequences. For each flow, it searches for a suitable GCD-like packet-size unit and derives four features:

1. Estimated GCD
2. Mean residual
3. Normalized mean residual
4. Fraction of packet sizes within the tolerance

The implementation searches candidate values from 8 to 256 bytes with a tolerance of 4 bytes.

## Models

### 1. No-IBNN

The baseline model uses the original normalized flow features without IBNN processing or Fuzzy-GCD feature replacement.

### 2. IBNN

The IBNN model uses the original flow features and an Information Bottleneck Neural Network.

### 3. Fuzzy-GCD

The Fuzzy-GCD model replaces selected flow features with normalized Fuzzy-GCD features before classification.

### 4. Fuzzy-GCD + IBNN

This model combines Fuzzy-GCD feature extraction with the IBNN architecture.

The prediction pipeline constructs both the standard feature representation and the GCD-enhanced representation before running the four models.

## Architecture

```text
Network Flow
     |
     v
Statistical Flow Features
     |
     +----------------------+
     |                      |
     v                      v
Original Features      Packet Size Sequence
     |                      |
     |                      v
     |                 Fuzzy-GCD
     |                      |
     |                 GCD Features
     |                      |
     +----------+-----------+
                |
                v
        Feature Representation
                |
       +--------+--------+
       |        |        |
       v        v        v
    No-IBNN   IBNN    Fuzzy-GCD
                         |
                         v
                  Fuzzy-GCD + IBNN
                         |
                         v
                    Prediction
```

## Dataset

The original dataset is intentionally excluded from the repository.

The evaluation pipeline combines two dataset sessions:

```text
VPN-nonVPN-Dataset/
└── Data/
    ├── session1/
    └── session2/
```

The two session flow files are loaded and combined before preprocessing.

Classes with fewer than 100 samples are removed before evaluation.

The remaining data is split into training and testing sets using a stratified 70/30 split with random seed 42.

## Features

The baseline feature vector contains 21 features:

```text
outer_bytes
outer_duration_ms
outer_first_matched_time_ms
outer_last_matched_time_ms
outer_capture_duration_ms
outer_packet_rate
outer_byte_rate
outer_bytes_in
outer_bytes_out
outer_min_piat_ms_in
outer_mean_piat_ms_in
outer_stddev_piat_ms_in
outer_max_piat_ms_in
outer_min_piat_ms_out
outer_mean_piat_ms_out
outer_stddev_piat_ms_out
outer_max_piat_ms_out
outer_min_piat_ms
outer_mean_piat_ms
outer_stddev_piat_ms
outer_max_piat_ms
```

The feature scaler and GCD scaler are loaded from the `models/` directory.

## Repository Structure

```text
Encrypted_network_flow_classifier/
│
├── backend/
│   ├── app.py                  # FastAPI server: models, pipeline, JSON API; mounts the web interface
│   ├── export_results.py       # builds backend/web/results.json from the CSVs below
│   ├── PythonNotebook.ipynb    # training and evaluation notebook (Kaggle)
│   ├── requirements.txt
│   ├── selected_flows/         # prepared test flows and evaluation outputs
│   │   ├── all_correct_*.parquet   # 9 flows all four models classify correctly
│   │   ├── gcd_only_*.parquet      # 6 flows only the GCD models classify correctly
│   │   ├── all_test_predictions.csv
│   │   ├── all_four_correct_selected.csv
│   │   ├── gcd_only_selected.csv
│   │   └── plots/              # overall_metrics.csv, gcd_label_metrics.csv and the PNG plots
│   └── web/                    # the web interface
│       ├── routes.py           # pages, HTMX fragments, CSV export, simulation event stream
│       ├── views.py            # everything the pages compute: insights, flow detail, slides
│       ├── icons.py            # inline SVG icons (lucide)
│       ├── results.json        # evaluation summary written by export_results.py
│       ├── templates/
│       │   ├── base.html       # layout: header, tabs, theme toggle, footer
│       │   ├── macros.html     # shared pieces: stat tiles, panels, bars, legends, slides
│       │   ├── dashboard.html  # page 1: hero and simulation
│       │   ├── classify.html   # page 2: upload and classify your own files
│       │   ├── project.html    # page 3: presentation of the project and results
│       │   └── partials/       # fragments: results, insights, flow table and detail, simulation cards
│       └── static/
│           ├── styles.css      # light and dark themes
│           ├── app.js          # browser behaviour: theme, tooltips, upload queue, simulation, slides
│           ├── favicon.svg
│           ├── img/            # the evaluation plots and the Fuzzy-GCD diagram
│           └── vendor/htmx.min.js
│
├── models/
│   ├── config.json
│   ├── feature_names.npy
│   ├── feature_scaler.pkl
│   ├── gcd_scaler.pkl
│   ├── label_encoder.pkl
│   ├── no_ibnn.pt
│   ├── ibnn.pt
│   ├── fuzzy_gcd.pt
│   └── fuzzy_gcd_ibnn.pt
│
├── analyse.py              # evaluation: predictions and selected example flows
├── plot.py                 # metric and GCD-improvement plots
├── predict_from_file.py    # classify one flow from a file on the command line
├── predict_one_flow.py     # classify one flow picked from the dataset
├── README.md
└── .gitignore
```

The repository does not include the original dataset or Python virtual environments. The prepared flows in `backend/selected_flows/` are included because the dashboard simulation uses them.

## Running the Application

Create a virtual environment and install the dependencies (from the repository root):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

Any of the environment folder names `.venv/`, `venv/`, `v/`, `env/` or `backend/.venv/` is ignored by git.

Start the server **from the repository root**, so that `backend.app` can find the `models/` directory:

```bash
python -m uvicorn backend.app:app --port 8000
```

Open <http://localhost:8000>. The same server provides the web interface and the JSON API, so there is nothing else to start. Add `--reload` to restart automatically when files under `backend/` change. Interactive API docs are at <http://localhost:8000/docs>.

To open the interface from another machine, add `--host 0.0.0.0` and browse to `http://<server-ip>:8000`.

### Pages

| Path | Page |
|---|---|
| `/` | Dashboard with the simulation |
| `/classify` | Upload and classify your own files |
| `/project` | Project presentation |

The pages also use a few internal routes (`/classify/run`, `/simulate`, `/runs/{id}/...`) that return HTML fragments, the simulation's event stream and CSV downloads. Results are kept in memory for the 20 most recent runs, so they are lost when the server restarts.

### API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Server status and the list of classes |
| `GET` | `/features` | The 21 feature names every flow must contain |
| `POST` | `/predict-file` | One file (`file`) containing exactly one flow |
| `POST` | `/predict-batch` | Several files (`files`), every row classified; up to 500 flows per request |
| `POST` | `/predict` | One flow as JSON: `{"flow": {...}}` |
| `GET` | `/samples` | The prepared flow files the dashboard simulation can run |
| `POST` | `/samples/{file_name}/classify` | Classify one prepared flow file by name |

Each classified flow includes the four model outputs with full class probabilities, the raw and scaled input features, the packet-size sequence, the Fuzzy-GCD features, both 5 × 5 input maps, and flow context (endpoints, protocol, TLS server name, packet count) when the file has those columns. The context is shown in the interface only; it is never a model input.

The prepared flows are read from `backend/selected_flows/`. Set `FLOWLENS_SAMPLES_DIR` to use another folder of single-flow Parquet files.

### Troubleshooting

| Symptom | Cause and fix |
|---|---|
| The page doesn't load | The server isn't running. Start it from the repository root and open <http://localhost:8000>. |
| "Address already in use" when starting the server | Another server is already on port 8000. Stop it (`pkill -f "uvicorn backend.app"`) or use `--port 8001` and open that port instead. |
| "These results have expired" | The server restarted or more than 20 newer runs replaced them. Run the classification or simulation again. |
| A file shows "Missing required columns" | The file lacks some of the 21 features or `outer_splt_ps`; see *Input files* below. |
| `ModuleNotFoundError` when starting the server | Install the requirements into the active environment: `pip install -r backend/requirements.txt`. |

## Using the Web Interface

The interface has three pages, switched from the top bar: `/`, `/classify` and `/project`.

### 1. Dashboard: simulation

**Run simulation** replays the prepared test flows from `backend/selected_flows/` through the models, one flow at a time. The server classifies each flow and streams the result to the page as it happens (server-sent events).

* **Scenario**: all prepared flows, only the ones all four models classify correctly, or only the ones where just the Fuzzy-GCD models are right.
* **Speed**: *Real time* steps through each inference stage (read, scale, Fuzzy-GCD, 5 × 5 maps, four models), *Fast* shortens the steps, and *Instant* classifies everything at once.
* While it runs: progress, the flow being classified and its current stage, live accuracy per model, and a live feed of each classified flow with its server name, endpoint and every model's verdict.
* When it finishes: the full results and insights below, with each model's accuracy on the whole 67,554-flow test set shown next to its accuracy on the sample.
* A summary of how the full test set splits between flows all models get right, flows only the GCD models get right, and flows all models miss.

### 2. Classify files

Upload your own flow files and get the same results and insights.

### Input files

The classifier accepts `.parquet` and `.csv` files. **Every row is one flow.** Each row needs:

| Column | Required | Used for |
|---|---|---|
| The 21 `outer_*` features listed above | Yes | Model input |
| `outer_splt_ps` | Yes | Packet-size sequence for Fuzzy-GCD (a list, or a string such as `"[60, 1500, 52]"`) |
| `flow_id` | No | Identifying the flow in the results |
| `application_name` | No | True label, used to mark predictions correct or wrong and compute accuracy |

The files in `backend/selected_flows/` are ready to upload. Any rows exported from the dataset's session flow files also work.

### Classifying

1. Drop files onto the upload area, or click it or **Choose files** to pick several at once.
2. Files collect in the **Queue**. Remove any you don't want; files that aren't Parquet or CSV are rejected with a note.
3. Click **Classify**. All queued files are sent in one request, and each file reports how many flows were read or why it failed (for example, missing columns).

### Results (dashboard and Classify files)

After classification both pages show batch-level insights:

* **Headline figures**: flows classified, how often all four models agree, mean confidence, the best model and majority-vote accuracy (when true labels are present).
* **Model scorecards**: accuracy, mean confidence, number of classes predicted, low-confidence predictions, and confidence when right versus wrong.
* **What each model predicts**: share of flows given each class, per model.
* **Model agreement**: a pairwise matrix of how often two models predict the same class.
* **Confidence distribution**: a histogram per model.
* **Fuzzy-GCD across flows**: the most common k values, mean residual and share of packets on the lattice.
* **Effect of Fuzzy-GCD features**: flows each GCD model fixes or breaks compared with its baseline, and flows where both GCD models are right while both baselines are wrong.
* **Most common mix-ups**: true label versus the class most models chose.
* **Every flow**: a table with each model's prediction, confidence, ✓/✕ against the true label and agreement, with filters (misclassified, models disagree, errors) and **Download CSV**.

The first flow opens automatically; select any row to switch. Each flow's breakdown shows:

* **Summary**: consensus class, mean confidence, verdict against the true label, packet count.
* **Pipeline**: the five inference stages with this flow's values.
* **Model cards**: top three classes, probability margin and uncertainty (normalised entropy) per model.
* **Probability comparison**: how the four models split probability across the leading classes.
* **Traffic profile**: duration, bytes, rates, inter-arrival time, packet sizes and the inbound/outbound byte split.
* **Packet sizes and fuzzy-GCD**: every packet as a bar, highlighted when it lies within 4 bytes of a multiple of k, plus the four GCD features.
* **What the models received**: all 21 features (raw and scaled), and the baseline and Fuzzy-GCD 5 × 5 input maps with the four swapped cells marked.

Hover or focus any chart mark for exact values.

### Light and dark mode

Use the sun/moon button in the top bar to switch themes. Dark is the default, and the choice is remembered in the browser. Each model keeps the same colour in both themes (No-IBNN blue, IBNN orange, Fuzzy-GCD aqua, Fuzzy-GCD + IBNN yellow); the palette is checked for colour-blind separation.

### 3. Project

A slide-style presentation of the work: the problem, dataset, pipeline, Fuzzy-GCD, the network architecture, the information bottleneck, the four experiments, overall and per-class results, where Fuzzy-GCD helps, agreement and errors, and conclusions. Use the side navigation, the floating controls, or the left and right arrow keys to move between slides.

Its figures come from `backend/web/results.json`. After re-running the evaluation, regenerate it (then restart the server):

```bash
python backend/export_results.py
```

## Prediction Scripts

Run both scripts from the repository root, with the backend's requirements installed; they import the models and preprocessing from `backend/app.py`.

### Predict From a File

Classifies a `.parquet` or `.csv` file containing exactly one flow and prints the flow, its GCD features and each model's prediction:

```bash
python predict_from_file.py backend/selected_flows/<flow>.parquet
```

### Predict a Single Flow

Loads `VPN-nonVPN-Dataset/Data/session1/session1_flows.parquet`, asks for a row index, and classifies that flow:

```bash
python predict_one_flow.py
```

## Evaluation

The evaluation pipeline:

1. Loads the dataset.
2. Filters classes based on minimum class count.
3. Performs a stratified train/test split.
4. Applies feature scaling.
5. Generates Fuzzy-GCD features.
6. Creates the standard and GCD-enhanced feature representations.
7. Runs all four models.
8. Converts predictions back to application labels.
9. Saves the predictions for further analysis.

Run it from the repository root with the dataset in place:

```bash
python analyse.py
```

Outputs in `selected_flows/`:

```text
all_test_predictions.csv       every test flow with all four predictions
all_four_correct_selected.csv  example flows all four models classify correctly
gcd_only_selected.csv          example flows only the GCD models classify correctly
*.parquet                      one file per selected flow, ready to upload in the web interface
```

## Evaluation Analysis

The project also identifies flows where:

* All four models correctly classify the flow.
* Both GCD-based models correctly classify the flow while both non-GCD models fail.

The second category is particularly useful for analyzing the contribution of Fuzzy-GCD features.

Distinct-label examples are selected and saved as CSV and Parquet files.

## Metrics and Visualization

The project includes visualization scripts for comparing model performance and analyzing labels that benefit from GCD features.

`plot.py` reads `selected_flows/all_test_predictions.csv` (so run `analyse.py` first) and writes to `selected_flows/plots/`:

```bash
python plot.py
```

```text
all_models_metrics.png
highest_gcd_labels.png
```

Copies of both plots are in `backend/web/static/img/` and shown on the **Project** page. The outputs used by the interface are kept in `backend/selected_flows/` (move `analyse.py`'s output there, or point `FLOWLENS_SAMPLES_DIR` at it).

The plots can be used to compare the four models and identify application labels where GCD-based features provide the largest improvement.

## Reproducibility

Training and evaluation are in `backend/PythonNotebook.ipynb`, written for Kaggle. It searches `DATA_ROOT` (default `/kaggle/input`) for the `session1_flows.parquet` and `session2_flows.parquet` files, so attach the dataset there or change `DATA_ROOT` to run it elsewhere. It uses:

```text
Random seed: 42
Test size: 30%
Minimum class count: 100
Epochs: 30 (AdamW, learning rate 1e-3, OneCycleLR, batch size 256)
IB beta sweep: 0.01, 0.005, 0.0001, 0.00005, 0.000001
```

The same preprocessing pipeline is used for all four models to make the model comparison consistent.

## Notes

The following are intentionally excluded from version control (see `.gitignore`):

```text
VPN-nonVPN-Dataset/        original dataset
/selected_flows/           analyse.py output at the repository root (backend/selected_flows/ is kept)
.venv/ venv/ v/ env/       Python virtual environments
backend/.venv/
__pycache__/               Python bytecode caches
.env, .env.*               local environment settings (.env.example is kept)
```

The trained model files and preprocessing artifacts are stored under `models/`.

## Technology Stack

### Machine Learning

* Python
* PyTorch
* Scikit-learn
* NumPy
* Pandas
* Joblib

### Backend

* Python
* FastAPI and Uvicorn
* PyArrow (Parquet input)
* Jinja2

### Web interface

* Jinja2 templates rendered by FastAPI
* HTMX for table filters and loading a flow's breakdown
* Server-sent events for the live simulation
* A small plain-JavaScript file for browser-only behaviour (theme, tooltips, drag and drop, slide keys)
* Plain CSS with switchable light and dark themes
* Lucide icons

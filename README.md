# RootSight — Auditable Root-Cause Analysis for Cloud Outage

---

## 1. What RootSight Is

**RootSight** is an evidence-based, deterministic root-cause analysis (RCA) engine and dashboard for cloud microservice incidents. It ranks candidate services from **metrics, structured logs, distributed traces, and dependency topology**; its rankings are evidence-based hypotheses, not proof of causation.

Unlike black-box AI chatbots that hallucinate causes, RootSight computes root causes using **reproducible, deterministic graph algorithms and robust statistical anomaly detection**. An optional Large Language Model (LLM) may translate the diagnosis into an executive briefing, but **the AI is strictly an explanation layer and never chooses, reorders, or invents the root-cause ranking**.

---

## 2. The Problem RootSight Solves

During cloud outages, Site Reliability Engineers (SREs) face three major challenges:
1. **Alert Cascades & Correlation Confusion**: A single database lock can trigger dozens of timeouts and error alerts in upstream caller services. Naive thresholding points fingers at the innocent victims instead of the true root cause.
2. **Untrustworthy "AI-Assisted" RCA**: LLMs applied directly to raw telemetry hallucinate non-existent database queries, misquote metric values, and invent phantom timestamps.
3. **Lack of Grounded Evidence**: Diagnostic recommendations lack verifiable links back to raw logs and metric records, preventing engineers from acting with confidence.

**RootSight solves this** by mathematically auditing every claim. Every cited fact is verified against the original telemetry records, ensuring 100% auditability.

---

## 3. How the System Works

RootSight executes a 7-stage deterministic diagnostic pipeline:

```text
       DATA INGESTION (Metrics, Logs, Traces, Topology)
                             ↓
         ANOMALY DETECTION (MAD Robust z-score & Fallback)
                             ↓
     SERVICE DEPENDENCY GRAPH (NetworkX Directed Graph)
                             ↓
         ROOT-CAUSE RANKING (Baseline Scoring & PageRank)
                             ↓
         EVIDENCE COLLECTION (Facts vs Inferences vs Actions)
                             ↓
      EVIDENCE VERIFICATION (Cryptographic-Style Record Audit)
                             ↓
 INTERACTIVE OBSERVABILITY DASHBOARD (Streamlit Dark Theme & REST API)
```

1. **Data Ingestion**: Ingests metrics (latency, error rate, CPU, memory, request volume), logs (severity, error count, message), traces (caller, callee, duration), and dependency relationships.
2. **Anomaly Detection**: Computes robust z-scores using median and Median Absolute Deviation (MAD). Handles zero variance safely and uses threshold fallbacks for sparse historical periods.
3. **Service Dependency Graph**: Builds a directed graph (`nx.DiGraph`) where $A \rightarrow B$ signifies service $A$ calls/depends on service $B$.
4. **Deterministic Ranking Engine**:
   - **Method A (Explainable Baseline)**: A weighted linear combination of metric anomaly severity $A(v)$, dependency propagation support $D(v)$, trace latency degradation $T(v)$, and log error surges $L(v)$.
   - **Method B (Personalized PageRank)**: Graph centrality propagation where random walks traverse call paths from symptomatic callers into the terminal bottleneck callee.
5. **Evidence Bundle Assembly**: Categorizes findings into **Observed Facts**, **Algorithmic Inferences**, and **Recommended Human Actions**.
6. **Evidence Verification**: Audits every cited identifier (`METRIC-*`, `LOG-*`, `TRACE-*`) against raw input files. Rejects phantom IDs.
7. **Interactive Dashboard**: Displays selected-dataset timeseries charts, directed service topologies, ranking breakdowns, and natural language executive briefings.

---

## 4. Architecture & Module Overview

```text
RootSight/
├── app.py                      # Main Streamlit Dark Observability Dashboard
├── api.py                      # FastAPI REST Backend Service
├── requirements.txt            # Production dependencies
├── .env.example                # Environment variable configuration template
├── .gitignore                  # Git hygiene rules
│
├── data/
│   ├── metrics.csv             # Raw timeseries metrics dataset
│   ├── logs.csv                # Raw structured log entries
│   ├── traces.csv              # Distributed trace spans
│   ├── dependencies.csv        # Service invocation topology
│   ├── incidents.json          # Demonstration incident metadata & ground truth
│   └── generate_datasets.py    # Deterministic telemetry generator script
│
├── engine/
│   ├── __init__.py
│   ├── data_loader.py          # Telemetry loading, schema validation & normalization
│   ├── graph_builder.py        # NetworkX directed graph topology builder
│   ├── ranking.py              # Method A (Baseline) & Method B (PageRank) ranking algorithms
│   └── pipeline.py             # End-to-end RCA execution orchestrator
│
├── detector/
│   ├── __init__.py
│   └── anomaly.py              # Robust z-score (MAD) & domain threshold detector
│
├── reporting/
│   ├── __init__.py
│   ├── evidence.py             # Evidence bundle collector & fact separator
│   ├── verifier.py             # Telemetry citation audit & verification engine
│   └── narrator.py             # Guarded narrative generator with offline fallback
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py              # Top-1, Top-3, and MRR accuracy calculations
│   └── benchmark.py            # Automated benchmark evaluation suite & RCAEval adapter
│
└── tests/                      # Automated unit and integration test suite (30 tests)
    ├── test_detector.py
    ├── test_graph_builder.py
    ├── test_ranking.py
    ├── test_evidence.py
    ├── test_verifier.py
    ├── test_evaluation.py
    └── test_pipeline.py
```

---

## 5. Technology Stack

- **Language**: Python 3.11+ (Fully compatible with Python 3.11, 3.12, 3.13, 3.14, 3.15)
- **Data Processing**: `pandas`, `numpy`
- **Topology Analysis**: `networkx`
- **Frontend Observability Dashboard**: `streamlit`
- **Interactive Visualizations**: `plotly`
- **REST API Backend**: `fastapi`, `uvicorn`, `pydantic`
- **Automated Testing**: `pytest`
- **Configuration Management**: `python-dotenv`

---

## 6. Installation & Quickstart

### Prerequisites
- Python 3.11 or higher installed on your system.
- Git (optional, for cloning).

### Step 1: Clone or Open the Repository
```bash
cd "c:\Users\shrey\OneDrive\Desktop\rv antygravity"
```

### Step 2: Set Up a Virtual Environment (Recommended)

#### On Windows:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### On macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 4: Configure Environment Variables (Optional)
RootSight is designed to work **100% offline without any external API keys**.
If you wish to experiment with optional LLM summaries:
```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```
Edit `.env` and add your `OPENAI_API_KEY` or `GEMINI_API_KEY`. If left empty, RootSight automatically uses its deterministic template narrator.

---

## 7. How to Start the Application

### Option A: Launch the Interactive Dashboard (Streamlit)
```bash
streamlit run app.py
```
Or if `streamlit` is not on your system PATH:
```bash
python -m streamlit run app.py
```
Open your browser to: **`http://localhost:8501`**

### Option B: Launch the FastAPI REST Service
```bash
python -m uvicorn api:app --reload --port 8765
```
Open interactive Swagger API docs at: **`http://localhost:8765/docs`**

---

## 8. Step-by-Step Demonstration Guide

Follow these exact steps to demonstrate RootSight:

1. **Select an Outage Scenario**:
   In the left sidebar, choose **`INC-001-DB-SPIKE — Database Latency Spike`**.
2. **Observe Instant Metric Cards**:
   - Monitored Services: `6`
   - Detected Anomalies: `3`
   - Inspect the computed leading candidate and score; these depend on the checked-in synthetic telemetry and current detector settings.
   - Evidence Verification: `VERIFIED` (All citations verified against raw telemetry)
3. **Inspect the Service Dependency Graph**:
   Click the **🌐 Service Dependency Graph** tab.
   - Hover over `postgres-db` (highlighted in red as Root Cause #1).
   - Observe that `cart-service` and `checkout-service` depend directly on `postgres-db`.
4. **Examine Telemetry Monitoring Charts**:
   Click the **📈 Telemetry Monitoring Charts** tab.
   - See the exact latency spike where `postgres-db` latency surged from 15ms to 950ms.
5. **Audit Evidence in Evidence Explorer**:
   Click the **🔎 Evidence Explorer** tab.
   - See the full candidate table ranking `postgres-db` #1.
   - Select `postgres-db` to inspect its ground-truth records: lock error logs (`LOG-PG-001`), latency metrics, and distributed traces.
6. **Compare Ranking Algorithms**:
   Switch the sidebar ranking method to **`Method B: Personalized PageRank`**.
   - Observe how PageRank propagates centrality along the graph topology.
7. **Run Built-in Benchmarking**:
   Click the **🏆 Evaluation & Benchmark** tab and click **"Run Full Benchmark Evaluation Across All Scenarios"**.
   - Observe Top-1 and Top-3 accuracy calculated deterministically across all test scenarios.

### Workspace navigation and persisted uploads

The Streamlit sidebar has dedicated destinations for Overview, Data Ingestion, Incident Analysis, Service Topology, Anomaly Detection, Root Cause Analysis, Evidence Explorer, Incident Timeline, Blast Radius, AI Investigation Assistant, and Reports / History. Data Ingestion accepts CSV/JSON files named for their type (metrics, logs, traces, dependencies), previews rows, validates required fields, and writes validated data to `data/rootsight.sqlite3`. Reuploading a dataset type for the same incident replaces its earlier batch. Analysis can use the persisted rows by selecting **Custom Telemetry Upload** and entering the same incident ID. Built-in scenarios are synthetic; there is no live telemetry connector.

The Streamlit ingestion, persisted telemetry retrieval, reset, and report-history controls call the FastAPI backend at `ROOTSIGHT_API_URL` (default `http://127.0.0.1:8765`). The API supports `POST /api/ingest` (JSON rows), `GET`/`DELETE /api/telemetry/{incident_id}`, and `GET /api/history`. `POST /api/analyze` persists its diagnosis in SQLite. Set `ROOTSIGHT_DB` to place the SQLite database elsewhere.

---

## 9. Built-in Incident Scenarios

RootSight includes 6 realistic cloud outage scenarios:

| Incident ID | Name | True Root Cause | Failure Mechanism |
| :--- | :--- | :--- | :--- |
| **`INC-001-DB-SPIKE`** | Database Latency Spike | `postgres-db` | Query lock contention on `order_items`, cascading timeouts to `cart-service` and `api-gateway`. |
| **`INC-002-PAY-FAIL`** | Payment Service Failure | `payment-service` | Expired TLS certificate throwing 500 internal errors; upstream `checkout-service` errors. |
| **`INC-003-API-GATEWAY`** | API Gateway Latency Spike | `api-gateway` | ReDoS regex route matching thread stall; backends remain healthy. |
| **`INC-004-INV-OVERLOAD`** | Inventory Service Overload | `inventory-service` | Flash sale CPU/RAM saturation (98% CPU, 94% RAM); high queue latency. |
| **`INC-005-HEALTHY`** | Normal Healthy System | `None` (Healthy) | Steady-state traffic; zero threshold violations; system reports healthy. |
| **`INC-006-INSUFFICIENT`**| Insufficient Evidence | `unknown` | Sparse metrics, missing traces; system explicitly reports inconclusive evidence. |

---

## 10. Running Automated Tests

Run the full pytest suite with verbose output:

```bash
python -m pytest -v
```

The test suite is in `tests/`. Run it in the same Python environment where dependencies are installed; results are environment-specific and are not asserted here.

## 11. Benchmark Evaluation & Metrics

RootSight provides an automated benchmark evaluation runner (`evaluation/benchmark.py`). Ground-truth labels are kept **strictly separate from the inputs to the diagnosis engine**.

### Accuracy Metrics
- **Top-1 Accuracy**: Fraction of cases where the true root cause ranks exactly #1:
  $$\text{Top-1} = \frac{\text{Cases with true root cause at Rank 1}}{\text{Total evaluated cases}}$$
- **Top-3 Accuracy**: Fraction of cases where the true root cause appears within the top 3 candidates:
  $$\text{Top-3} = \frac{\text{Cases with true root cause in Top 3}}{\text{Total evaluated cases}}$$
- **Mean Reciprocal Rank (MRR)**: Average reciprocal rank of the ground-truth service:
  $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

Benchmark output is computed at runtime by `/api/benchmark` or the dashboard; no precomputed accuracy figures are asserted here.

### RCAEval Dataset Support
RootSight includes a built-in schema adapter (`BenchmarkRunner.load_rcaeval_incident`) that maps RCAEval benchmark fields (`cmdb_id`, `kpi_name`, `value`, `timestamp`) into RootSight's telemetry schema. RCAEval results are explicitly labelled and distinguished from built-in synthetic scenarios.

---

## 12. Mathematical Formulation of Ranking Algorithms

### Method A: Explainable Baseline
The candidate score $S(v)$ for service $v$ is computed as:

$$S(v) = w_A \cdot A(v) + w_D \cdot D(v) + w_T \cdot T(v) + w_L \cdot L(v)$$

Where:
- $A(v) \in [0, 1]$: Metric anomaly severity from robust z-score:
  $$\text{robust\_z} = \frac{0.6745 \cdot (x - \text{median}(X))}{\text{MAD}(X)}$$
- $D(v) \in [0, 1]$: Dependency propagation support. Boosted if upstream callers exhibit anomalies; this relationship is observational and does not establish causality.
- $T(v) \in [0, 1]$: Trace latency anomaly score when $v$ is the invoked callee.
- $L(v) \in [0, 1]$: Normalized error-level log frequency.
- Default weights: $w_A = 0.35, w_D = 0.30, w_T = 0.20, w_L = 0.15$ (normalized to $\sum w = 1.0$).

### Method B: Personalized PageRank
Personalized PageRank computes the stationary distribution of a random walk with restart over the service dependency graph $G = (V, E)$:

$$\mathbf{p} = (1 - \alpha) \mathbf{v}_{\text{pers}} + \alpha \mathbf{W}^T \mathbf{p}$$

- **Graph Direction**: Edges $u \rightarrow v$ represent service $u$ calling $v$.
- **Edge Weighting**: Transitions into callee $v$ are weighted by $v$'s anomaly evidence $W_{uv} = W_{uv}^{(0)} \cdot (1 + 4 \cdot A(v))$.
- **Sink Handling**: Self-loops are added to sink nodes (services with out-degree 0, like databases) so random walk probability concentrates at the bottleneck callee.
- **Personalization Vector**: $\mathbf{v}_{\text{pers}}(v) \propto 0.45 A(v) + 0.35 T(v) + 0.20 L(v)$.

---

## 13. Security and Reliability Guardrails

- **Zero API Key Leakage**: API keys are read strictly from environment variables (`.env`). No keys are committed to source control or exposed in the UI.
- **Deterministic Independence**: The ranking algorithm functions 100% deterministically without requiring an LLM or network connection.
- **Cryptographic Auditability**: The evidence verifier checks all cited records against raw inputs, detecting and rejecting hallucinations.
- **Safe Remediation Boundary**: RootSight **never** automatically executes remediation actions on cloud infrastructure. All recommendations are explicitly marked for human engineer review.

---

## 14. Troubleshooting & FAQ

**Q: Streamlit command is not recognized.**
*A: Run `python -m streamlit run app.py` instead of `streamlit run app.py`.*

**Q: Port 8501 is already in use.**
*A: Specify a custom port: `python -m streamlit run app.py --server.port 8502`.*

**Q: How do I load my own custom telemetry CSV?**
*A: In the sidebar, switch the Data Source Mode to "Custom Telemetry Upload (CSV/JSON)" and upload your metrics, logs, and traces files. RootSight will validate the schema and report any missing columns.*

**Q: Can RootSight run without an OpenAI or Gemini API key?**
*A: Yes! RootSight operates completely offline by default. It uses a built-in deterministic template narrator to generate structured incident briefings.*

---

## 15. License

RootSight is released under the Apache 2.0 Open Source License.

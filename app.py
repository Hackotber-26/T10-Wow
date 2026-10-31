"""
RootSight - Auditable Root-Cause Analysis for Cloud Outages
Main Streamlit Observability Dashboard.
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import requests

from engine.data_loader import DataLoader
from engine.pipeline import DiagnosisPipeline, PipelineResult
from evaluation.benchmark import BenchmarkRunner

API_BASE_URL = os.environ.get("ROOTSIGHT_API_URL", "http://127.0.0.1:8765").rstrip("/")


def frame_records(frame):
    if frame is None or frame.empty:
        return []
    return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")

# Page Configuration
st.set_page_config(
    page_title="RootSight — Cloud Outage Root-Cause Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-Contrast Dark Observability Theme CSS
st.markdown("""
<style>
    /* Global Background & Typography */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #f8fafc !important;
        font-weight: 600;
        letter-spacing: -0.02em;
    }

    /* Top Navigation Banner */
    .top-nav {
        background: #111827;
        border: 1px solid #1f293d;
        border-radius: 8px;
        padding: 14px 22px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .brand-title {
        font-size: 1.45rem;
        font-weight: 700;
        color: #38bdf8;
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }
    .brand-tagline {
        font-size: 0.82rem;
        color: #94a3b8;
        margin-left: 12px;
    }

    /* Metric Cards */
    .metric-card {
        background: #111827;
        border: 1px solid #1f293d;
        border-radius: 8px;
        padding: 14px 18px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }
    .metric-val {
        font-size: 1.55rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-top: 4px;
    }
    .metric-sub {
        font-size: 0.72rem;
        color: #64748b;
        margin-top: 2px;
    }

    /* Badges */
    .badge-critical {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.75rem;
    }
    .badge-warning {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.75rem;
    }
    .badge-healthy {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.75rem;
    }
    .badge-verified {
        background: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 3px 10px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.8rem;
    }

    /* Container borders */
    [data-testid="stSidebar"] {
        background-color: #0d121f;
        border-right: 1px solid #1f293d;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        border-radius: 7px;
        padding: 7px 9px;
        color: #cbd5e1 !important;
        transition: background-color 120ms ease, color 120ms ease;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label p,
    [data-testid="stSidebar"] [role="radiogroup"] label span {
        color: inherit !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background-color: #172235;
        color: #f8fafc !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:focus-within {
        outline: 2px solid #38bdf8;
        outline-offset: 1px;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background-color: #172235;
        color: #f8fafc !important;
        box-shadow: inset 3px 0 0 #38bdf8;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #1f293d;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #111827;
        border: 1px solid #1f293d;
        border-radius: 6px 6px 0 0;
        color: #94a3b8;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
        border-bottom: 2px solid #38bdf8 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_loader():
    return DataLoader()


@st.cache_resource
def get_pipeline(mad_thresh: float, min_obs: int):
    return DiagnosisPipeline(mad_threshold=mad_thresh, min_history_observations=min_obs)


loader = get_loader()
incidents_meta = loader.load_incidents_metadata()

# Top Header / Brand
st.markdown("""
<div class="top-nav">
    <div>
        <span class="brand-title">🔍 RootSight</span>
        <span class="brand-tagline">Auditable Root-Cause Analysis for Cloud Outages</span>
    </div>
    <div>
        <span style="font-size: 0.8rem; color: #64748b; margin-right: 10px;">Deterministic Telemetry Engine</span>
        <span class="badge-healthy">ENGINE READY</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.markdown("### ⚙️ Incident & Telemetry Controls")

workspace_page = st.sidebar.radio("Workspace", [
    "Overview Dashboard", "Data Ingestion", "Incident Analysis", "Service Topology",
    "Anomaly Detection", "Root Cause Analysis", "Evidence Explorer", "Incident Timeline",
    "Blast Radius", "AI Investigation Assistant", "Reports / History",
], key="workspace_page")

data_mode = st.sidebar.radio(
    "Data Source Mode",
    ["Built-in Demonstration Scenarios", "Custom Telemetry Upload (CSV/JSON)"],
    index=0,
)

selected_incident_id = "INC-001-DB-SPIKE"
uploaded_metrics = None
uploaded_logs = None
uploaded_traces = None
uploaded_deps = None

if data_mode == "Built-in Demonstration Scenarios":
    scenario_options = {
        inc["incident_id"]: f"{inc['incident_id']} — {inc['name']}" for inc in incidents_meta
    }
    selected_incident_id = st.sidebar.selectbox(
        "Select Outage Incident Scenario",
        list(scenario_options.keys()),
        format_func=lambda x: scenario_options[x],
        index=0,
    )
    inc_info = next((i for i in incidents_meta if i["incident_id"] == selected_incident_id), None)
    if inc_info:
        st.sidebar.caption(f"**Context:** {inc_info.get('description', '')}")
        st.sidebar.caption(f"**Category:** `{inc_info.get('category', 'unknown')}`")
else:
    st.sidebar.markdown("#### Upload Custom Telemetry")
    selected_incident_id = st.sidebar.text_input("Uploaded incident ID", value="uploaded-incident")
    uploaded_metrics = st.sidebar.file_uploader("Upload Metrics CSV/JSON", type=["csv", "json"])
    uploaded_logs = st.sidebar.file_uploader("Upload Logs CSV/JSON", type=["csv", "json"])
    uploaded_traces = st.sidebar.file_uploader("Upload Traces CSV/JSON", type=["csv", "json"])
    uploaded_deps = st.sidebar.file_uploader("Upload Topology CSV/JSON (Optional)", type=["csv", "json"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧮 Algorithmic Tuning")

ranking_method = st.sidebar.selectbox(
    "Root-Cause Ranking Method",
    [
        "Method A: Explainable Baseline (Weighted Signals)",
        "Method B: Personalized PageRank (Graph Centrality)",
    ],
    index=0,
)
method_key = "baseline" if "Baseline" in ranking_method else "pagerank"

mad_threshold = st.sidebar.slider(
    "Robust z-score MAD Threshold",
    min_value=1.5,
    max_value=6.0,
    value=3.0,
    step=0.5,
    help="Flags metrics where |0.6745 * (value - median) / MAD| exceeds this threshold.",
)

min_history = st.sidebar.number_input(
    "Min Historical Observations",
    min_value=2,
    max_value=20,
    value=4,
    help="Minimum timeseries observations required before engaging MAD robust statistics vs threshold fallback.",
)

report_mode = st.sidebar.radio(
    "Report Generation Engine",
    ["Deterministic Template (Guaranteed Offline)", "Guarded AI Narration (Optional API)"],
    index=0,
)
prefer_ai = ("AI" in report_mode)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reload & Re-analyze Telemetry", use_container_width=True):
    st.rerun()

# Data Ingestion Logic
pipeline = get_pipeline(mad_thresh=mad_threshold, min_obs=min_history)

if data_mode == "Built-in Demonstration Scenarios":
    df_metrics, df_logs, df_traces, df_deps = loader.load_incident_telemetry(selected_incident_id)
else:
    # Uploaded files take precedence; otherwise reuse the validated persisted dataset.
    try:
        persisted_response = requests.get(f"{API_BASE_URL}/api/telemetry/{selected_incident_id}", timeout=3)
        persisted = persisted_response.json().get("datasets", {}) if persisted_response.status_code == 200 else {}
        if persisted_response.status_code not in (200, 404):
            st.sidebar.error(f"Telemetry API returned {persisted_response.status_code}: {persisted_response.text[:180]}")
    except requests.RequestException as exc:
        persisted = {}
        st.sidebar.warning(f"Persisted uploads unavailable: cannot reach API at {API_BASE_URL} ({exc}).")
    def read_upload(upload, kind):
        if upload is None: return pd.DataFrame(persisted.get(kind, []))
        try:
            raw = upload.getvalue()
            frame = pd.read_csv(__import__("io").BytesIO(raw)) if upload.name.lower().endswith(".csv") else pd.DataFrame(json.loads(raw))
            check = {"metrics": loader.validate_metrics_df, "logs": loader.validate_logs_df, "traces": loader.validate_traces_df, "dependencies": loader.validate_dependencies_df}.get(kind)
            if check:
                checked = check(frame)
                if not checked.is_valid:
                    st.sidebar.error(f"{kind}: {'; '.join(checked.errors)}"); return pd.DataFrame()
                return checked.cleaned_df
            return frame
        except Exception as exc:
            st.sidebar.error(f"Could not parse {kind}: {exc}"); return pd.DataFrame()
    df_metrics = read_upload(uploaded_metrics, "metrics")
    df_logs = read_upload(uploaded_logs, "logs")
    df_traces = read_upload(uploaded_traces, "traces")
    df_deps = read_upload(uploaded_deps, "dependencies")

# Run RCA Pipeline
with st.spinner("Analyzing telemetry: detecting anomalies, constructing dependency topology, and ranking root causes..."):
    result: PipelineResult = pipeline.run(
        incident_id=selected_incident_id,
        df_metrics=df_metrics,
        df_logs=df_logs,
        df_traces=df_traces,
        df_dependencies=df_deps,
        ranking_method=method_key,
        prefer_ai_narration=prefer_ai,
    )

# Dedicated workspace pages. Data lineage is always labelled; the built-in scenario is synthetic.
st.markdown(f"## {workspace_page}")
if data_mode == "Built-in Demonstration Scenarios":
    st.caption("Data source: **synthetic demonstration telemetry**. No live monitoring integration is configured.")
else:
    st.caption("Data source: **uploaded telemetry**. This app does not claim live monitoring.")

if workspace_page == "Data Ingestion":
    st.write("Upload one or more CSV or JSON telemetry files. Preview and validate before saving through the RootSight API to SQLite.")
    ingestion_id = st.text_input("Incident ID for this upload", value="uploaded-incident")
    files = st.file_uploader("Drop metrics, logs, traces, or dependencies here", type=["csv", "json"], accept_multiple_files=True, key="ingestion_files")
    parsed = {}
    for f in files or []:
        kind = next((k for k in ("metrics", "logs", "traces", "dependencies") if k in f.name.lower()), None)
        if not kind:
            st.warning(f"{f.name}: include metrics, logs, traces, or dependencies in the filename to map its dataset type.")
            continue
        try:
            raw = f.getvalue()
            frame = pd.read_csv(__import__("io").BytesIO(raw)) if f.name.lower().endswith(".csv") else pd.DataFrame(json.loads(raw))
            validator = {"metrics": loader.validate_metrics_df, "logs": loader.validate_logs_df, "traces": loader.validate_traces_df, "dependencies": loader.validate_dependencies_df}.get(kind)
            if validator:
                report = validator(frame)
                if not report.is_valid:
                    st.error(f"{f.name}: " + "; ".join(report.errors)); continue
                frame = report.cleaned_df
                for warning in report.warnings: st.warning(f"{f.name}: {warning}")
            parsed[kind] = frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")
            st.success(f"{f.name}: {len(frame)} valid rows")
            st.dataframe(frame.head(8), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Could not parse {f.name}: {exc}")
    if parsed and st.button("Save validated telemetry", type="primary"):
        try:
            response = requests.post(f"{API_BASE_URL}/api/ingest", json={"incident_id": ingestion_id.strip(), **parsed}, timeout=20)
            if response.ok:
                confirmation = response.json()
                st.success(f"API persisted {confirmation['rows_saved']} rows for {ingestion_id}.")
                for kind, warnings in confirmation.get("warnings", {}).items():
                    for warning in warnings: st.warning(f"{kind}: {warning}")
            else:
                st.error(f"API rejected telemetry ({response.status_code}): {response.text[:300]}")
        except requests.RequestException as exc:
            st.error(f"Telemetry was not saved: API at {API_BASE_URL} is unavailable ({exc}).")
    try:
        saved_response = requests.get(f"{API_BASE_URL}/api/telemetry/{ingestion_id.strip()}", timeout=3)
        saved = saved_response.json().get("datasets", {}) if saved_response.status_code == 200 else {k: [] for k in ("metrics", "logs", "traces", "dependencies")}
        if saved_response.status_code not in (200, 404): st.error(f"Could not verify saved telemetry ({saved_response.status_code}): {saved_response.text[:200]}")
    except requests.RequestException as exc:
        saved = {k: [] for k in ("metrics", "logs", "traces", "dependencies")}
        st.warning(f"Could not verify persistence: API at {API_BASE_URL} is unavailable ({exc}).")
    if any(saved.values()):
        st.caption("Previously persisted rows for this incident")
        st.json({kind: len(rows) for kind, rows in saved.items()})
    if st.button("Reset uploaded telemetry for this incident"):
        try:
            reset_response = requests.delete(f"{API_BASE_URL}/api/telemetry/{ingestion_id.strip()}", timeout=10)
            if reset_response.ok: st.info(f"Removed {reset_response.json()['deleted_batches']} persisted telemetry batches.")
            else: st.error(f"Reset failed ({reset_response.status_code}): {reset_response.text[:200]}")
        except requests.RequestException as exc:
            st.error(f"Reset failed: API at {API_BASE_URL} is unavailable ({exc}).")
elif workspace_page == "Incident Analysis":
    st.write(result.report.summary)
    st.caption("This diagnosis is calculated from the selected dataset using deterministic analytics.")
    st.dataframe(pd.DataFrame([c.to_dict() for c in result.candidates]), use_container_width=True, hide_index=True)
elif workspace_page == "Service Topology":
    st.write("Observed service call/dependency graph. Edges describe relationships and do not prove causation.")
    graph = result.graph_result.graph
    if graph.number_of_nodes():
        positions = nx.circular_layout(graph, scale=1)
        ex, ey = [], []
        for src, dst in graph.edges():
            ex.extend([positions[src][0], positions[dst][0], None]); ey.extend([positions[src][1], positions[dst][1], None])
        edge_trace = go.Scatter(x=ex, y=ey, mode="lines", line={"color": "#475569", "width": 1.5}, hoverinfo="skip")
        nodes = list(graph.nodes())
        colors = ["#ef4444" if s == (result.top_candidate.service if result.top_candidate else None) else "#f59e0b" if result.graph_result.nodes.get(s) and result.graph_result.nodes[s].is_anomalous else "#10b981" for s in nodes]
        node_hover = []
        for service in nodes:
            info = result.graph_result.nodes.get(service)
            candidate = next((c for c in result.candidates if c.service == service), None)
            node_hover.append(f"<b>{service}</b><br>Status: {'anomalous' if info and info.is_anomalous else 'no anomaly observed'}<br>Callers: {', '.join(info.callers) if info and info.callers else 'None'}<br>Dependencies: {', '.join(info.dependencies) if info and info.dependencies else 'None'}<br>RCA score: {candidate.score:.3f}" if candidate else f"<b>{service}</b><br>Status: {'anomalous' if info and info.is_anomalous else 'no anomaly observed'}<br>No candidate score")
        node_trace = go.Scatter(x=[positions[n][0] for n in nodes], y=[positions[n][1] for n in nodes], mode="markers+text", text=nodes, textposition="top center", hovertext=node_hover, hovertemplate="%{hovertext}<extra></extra>", marker={"size": 22, "color": colors, "line": {"width": 1, "color": "#0b0f19"}})
        arrows = [dict(x=positions[dst][0], y=positions[dst][1], ax=positions[src][0], ay=positions[src][1], xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=1.2, arrowcolor="#64748b", standoff=13) for src, dst in graph.edges()]
        fig = go.Figure([edge_trace, node_trace]); fig.update_layout(template="plotly_dark", paper_bgcolor="#0d121f", plot_bgcolor="#0d121f", font_color="#e2e8f0", height=540, margin={"l": 32, "r": 32, "t": 40, "b": 35}, showlegend=False, annotations=arrows, xaxis={"visible": False}, yaxis={"visible": False})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No dependency relationships were observed in this dataset.")
elif workspace_page == "Anomaly Detection":
    st.write("Robust MAD and domain threshold detector output; empty evidence is not treated as a healthy baseline.")
    st.dataframe(pd.DataFrame([a.to_dict() for a in result.anomalies]), use_container_width=True, hide_index=True)
elif workspace_page == "Root Cause Analysis":
    st.write(result.report.ranking_rationale)
    st.dataframe(pd.DataFrame([{"rank": c.rank, "service": c.service, "score": c.score, "metric": c.metric_signal, "dependency": c.dependency_signal, "trace": c.trace_signal, "logs": c.log_signal, "evidence": c.supporting_evidence_ids} for c in result.candidates]), use_container_width=True, hide_index=True)
    st.caption("Scores are deterministic evidence rankings. Dependency and chronology signals indicate support, not proof of causation.")
elif workspace_page == "Evidence Explorer":
    for service, bundle in result.bundles.items():
        with st.expander(service):
            st.write("Metrics"); st.dataframe(pd.DataFrame(bundle.metrics_summary), use_container_width=True, hide_index=True)
            st.write("Logs"); st.dataframe(pd.DataFrame(bundle.logs_summary), use_container_width=True, hide_index=True)
            st.write("Traces"); st.dataframe(pd.DataFrame(bundle.traces_summary), use_container_width=True, hide_index=True)
elif workspace_page == "Incident Timeline":
    events = []
    for kind, frame, label_col in (("metric", df_metrics, "metric_name"), ("log", df_logs, "level"), ("trace", df_traces, "trace_id")):
        if frame is not None and not frame.empty:
            for _, row in frame.iterrows():
                events.append({"timestamp": row.get("timestamp", "unknown"), "type": kind, "service": row.get("service", row.get("destination_service", row.get("destination", "unknown"))), "event": row.get(label_col, "telemetry"), "value/message": row.get("metric_value", row.get("message", row.get("latency_ms", "")))})
    if events:
        timeline = pd.DataFrame(events).sort_values("timestamp", kind="stable")
        st.dataframe(timeline, use_container_width=True, hide_index=True)
    else: st.info("No timestamped telemetry is available for this incident.")
elif workspace_page == "Blast Radius":
    graph = result.graph_result.graph
    root = result.top_candidate.service if result.top_candidate else None
    if root and root in graph:
        observed = sorted(graph.predecessors(root))
        predicted = sorted(n for n in nx.ancestors(graph, root) if n not in observed)
        st.metric("Direct observed callers", len(observed)); st.write("Observed:", observed or "None")
        st.write("Potential upstream impact by transitive dependency:", predicted or "None")
        st.caption("Potential impact is a graph projection. It is not evidence that these services failed.")
    else: st.info("No ranked service is available to calculate impact.")
    st.markdown("### What-if failure simulator")
    st.caption("Hypothetical projection from the observed dependency graph; it does not create or record an incident.")
    if graph.number_of_nodes():
        hypothetical_service = st.selectbox("Assume this service fails", sorted(graph.nodes), key="what_if_failed_service")
        hypothetical_callers = sorted(nx.ancestors(graph, hypothetical_service))
        st.metric("Potentially affected callers", len(hypothetical_callers))
        st.write(hypothetical_callers or "No upstream callers are represented in this graph.")
elif workspace_page == "AI Investigation Assistant":
    st.caption("Evidence-grounded helper. AI narration is optional; the deterministic ranking above remains authoritative.")
    question = st.text_input("Ask about this incident", placeholder="Why is this service ranked first?")
    if not question or any(term in question.lower() for term in ("why", "rank", "cause", "score")):
        st.write(result.report.ranking_rationale)
    elif any(term in question.lower() for term in ("impact", "blast", "affected")):
        st.write("Observed direct callers:", result.report.affected_services or "None listed")
        st.caption("These are observed graph relationships; they do not establish causality or confirm service failure.")
    elif any(term in question.lower() for term in ("evidence", "source", "citation", "log", "trace")) and result.top_candidate:
        bundle = result.bundles.get(result.top_candidate.service)
        st.write("Evidence IDs:", ", ".join(bundle.cited_evidence_ids) if bundle and bundle.cited_evidence_ids else "No cited source records")
    else:
        st.write(result.report.summary)
    for i, suggestion in enumerate(result.report.suggested_next_steps, 1): st.markdown(f"{i}. {suggestion}")
    if result.top_candidate:
        bundle = result.bundles.get(result.top_candidate.service)
        if bundle:
            st.write("Source evidence IDs:", ", ".join(bundle.cited_evidence_ids) if bundle.cited_evidence_ids else "No evidence IDs available")
            with st.expander(f"Inspect {result.top_candidate.service} source records"):
                st.dataframe(pd.DataFrame(bundle.metrics_summary), use_container_width=True, hide_index=True)
                st.dataframe(pd.DataFrame(bundle.logs_summary), use_container_width=True, hide_index=True)
                st.dataframe(pd.DataFrame(bundle.traces_summary), use_container_width=True, hide_index=True)
elif workspace_page == "Reports / History":
    timeline_rows = []
    for kind, frame in (("metric", df_metrics), ("log", df_logs), ("trace", df_traces)):
        if frame is not None and not frame.empty:
            for _, row in frame.iterrows():
                timeline_rows.append({"timestamp": row.get("timestamp", "unknown"), "type": kind, "service": row.get("service", row.get("destination_service", row.get("destination", "unknown"))), "record": row.to_dict()})
    timeline_rows.sort(key=lambda e: str(e["timestamp"]))
    report_payload = {"incident_id": result.incident_id, "source": "synthetic" if data_mode.startswith("Built-in") else "uploaded", "summary": result.report.to_dict(), "rankings": [c.to_dict() for c in result.candidates], "evidence": {service: bundle.to_dict() for service, bundle in result.bundles.items()}, "verification": result.verification_result.to_dict(), "timeline": timeline_rows, "limitations": result.report.limitations_and_uncertainty}
    st.download_button("Export incident report (JSON)", json.dumps(report_payload, indent=2, default=str), file_name=f"{result.incident_id}-report.json", mime="application/json")
    if st.button("Save analysis to history"):
        try:
            analysis_response = requests.post(f"{API_BASE_URL}/api/analyze", json={"incident_id": result.incident_id, "metrics": frame_records(df_metrics), "logs": frame_records(df_logs), "traces": frame_records(df_traces), "dependencies": frame_records(df_deps)}, timeout=20)
            if analysis_response.ok: st.success(f"Saved analysis for {result.incident_id} through the API.")
            else: st.error(f"Analysis was not saved ({analysis_response.status_code}): {analysis_response.text[:250]}")
        except requests.RequestException as exc:
            st.error(f"Analysis was not saved: API at {API_BASE_URL} is unavailable ({exc}).")
    try:
        history_response = requests.get(f"{API_BASE_URL}/api/history", timeout=5)
        if history_response.ok:
            history = history_response.json().get("analyses", [])
            st.dataframe(pd.DataFrame([{k: row[k] for k in ("id", "incident_id", "created_at")} for row in history]), use_container_width=True, hide_index=True)
        else: st.error(f"Could not load history ({history_response.status_code}): {history_response.text[:200]}")
    except requests.RequestException as exc:
        st.warning(f"Analysis history unavailable: API at {API_BASE_URL} could not be reached ({exc}).")

# SECTION C: INCIDENT SUMMARY METRIC CARDS
top_cand = result.top_candidate
top_svc = top_cand.service if top_cand else "None"
top_score = top_cand.score if top_cand else 0.0
total_anoms = len(result.anomalies)
total_services = len(result.services)
audit_status = result.verification_result.audit_status

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Monitored Services</div>
        <div class="metric-val">{total_services}</div>
        <div class="metric-sub">Topology nodes</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Detected Anomalies</div>
        <div class="metric-val" style="color: {'#ef4444' if total_anoms > 0 else '#10b981'};">{total_anoms}</div>
        <div class="metric-sub">Across metrics, logs, traces</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Ranked Candidates</div>
        <div class="metric-val">{len(result.candidates)}</div>
        <div class="metric-sub">Deterministic evaluation</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    badge_color = "#ef4444" if top_score >= 0.70 else ("#f59e0b" if top_score >= 0.30 else "#10b981")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Top Root Cause</div>
        <div class="metric-val" style="color: {badge_color}; font-size: 1.25rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{top_svc}</div>
        <div class="metric-sub">Score: {top_score:.2f} ({top_cand.evidence_quality if top_cand else 'N/A'})</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    v_class = "badge-healthy" if result.verification_result.passed else "badge-critical"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Evidence Verification</div>
        <div class="metric-val" style="font-size: 1.1rem; color: {'#34d399' if result.verification_result.passed else '#f87171'};">{audit_status}</div>
        <div class="metric-sub">{result.verification_result.passed_count}/{result.verification_result.total_checked} telemetry citations audited</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# Main Multi-Tab Investigation Workspace
tab_diagnosis, tab_graph, tab_charts, tab_evidence, tab_benchmark = st.tabs([
    "📋 Diagnosis & Narration",
    "🌐 Service Dependency Graph",
    "📈 Telemetry Monitoring Charts",
    "🔎 Evidence Explorer",
    "🏆 Evaluation & Benchmark",
])

# TAB 1: DIAGNOSIS & NARRATION
with tab_diagnosis:
    st.markdown("### 📋 Outage Root-Cause Diagnosis Report")
    report = result.report

    # Header Alert Box
    if report.leading_candidate in ["None", "None (Healthy or Inconclusive)"]:
        st.success(f"✅ **System Healthy:** {report.summary}")
    else:
        st.error(f"🚨 **Leading Root Cause:** `{report.leading_candidate}` — Confidence: **{report.confidence_level}** | Method: `{result.method_used}`")

    diag_col1, diag_col2 = st.columns([3, 2])

    with diag_col1:
        st.markdown("#### 1. Executive Summary")
        st.write(report.summary)

        st.markdown("#### 2. Deterministic Ranking Rationale")
        st.info(report.ranking_rationale)

        st.markdown("#### 3. Grounded Telemetry Evidence")
        for fact in report.supporting_evidence_facts:
            st.markdown(fact)

        st.markdown("#### 4. Limitations & Uncertainty")
        st.caption(report.limitations_and_uncertainty)

    with diag_col2:
        st.markdown("#### 5. Recommended Investigation Actions")
        for step in report.suggested_next_steps:
            st.markdown(f"- 🛠️ {step}")

        st.markdown("#### 6. Impacted Upstream Callers")
        for svc in report.affected_services:
            st.markdown(f"- ⚠️ `{svc}`")

        st.markdown("#### 7. Alternative Candidates")
        for alt in report.alternative_candidates:
            st.markdown(f"- 🔸 {alt}")

        st.markdown("---")
        st.markdown(f"**Narrator Engine:** `{report.narrator_type}`")
        st.markdown(f"**Verification Audit Status:** `{report.verification_badge}`")

# TAB 2: INTERACTIVE DEPENDENCY GRAPH
with tab_graph:
    st.markdown("### 🌐 Service Dependency Topology")
    st.caption("Directed calls: source -> destination. Anomalous nodes and root-cause bottleneck highlighted.")

    G = result.graph_result.graph
    if len(G.nodes) > 0:
        pos = nx.spring_layout(G, seed=42, k=0.9)

        edge_x = []
        edge_y = []
        edge_text = []
        for u, v, d in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1.5, color="#475569"),
            hoverinfo="none",
            mode="lines",
        )

        node_x = []
        node_y = []
        node_colors = []
        node_text = []
        node_sizes = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            n_info = result.graph_result.nodes.get(node)
            is_root = (node == top_svc and top_score >= 0.20)
            is_anom = n_info.is_anomalous if n_info else False

            if is_root:
                color = "#ef4444"  # Red for root cause
                size = 32
                status = "ROOT CAUSE (#1)"
            elif is_anom:
                color = "#f59e0b"  # Amber for anomalous
                size = 24
                status = "ANOMALOUS"
            else:
                color = "#10b981"  # Green for healthy
                size = 20
                status = "HEALTHY"

            node_colors.append(color)
            node_sizes.append(size)

            callers_str = ", ".join(n_info.callers) if n_info and n_info.callers else "None"
            deps_str = ", ".join(n_info.dependencies) if n_info and n_info.dependencies else "None"
            hover = (
                f"<b>Service:</b> {node}<br>"
                f"<b>Status:</b> {status}<br>"
                f"<b>Callers (Upstream):</b> {callers_str}<br>"
                f"<b>Dependencies (Downstream):</b> {deps_str}<br>"
                f"<b>In-Degree:</b> {n_info.in_degree if n_info else 0} | <b>Out-Degree:</b> {n_info.out_degree if n_info else 0}"
            )
            node_text.append(hover)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=[node for node in G.nodes()],
            textposition="top center",
            textfont=dict(color="#f8fafc", size=11),
            hoverinfo="text",
            hovertext=node_text,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color="#0b0f19"),
            ),
        )

        fig_graph = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                plot_bgcolor="#0d121f",
                paper_bgcolor="#0d121f",
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=20, r=20, t=20),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=520,
            ),
        )
        st.plotly_chart(fig_graph, use_container_width=True)
    else:
        st.warning("No graph nodes available for this dataset.")

# TAB 3: TELEMETRY MONITORING CHARTS
with tab_charts:
    st.markdown("### 📈 Telemetry Signals for Selected Dataset")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### Service Latency (ms)")
        if df_metrics is not None and not df_metrics.empty and "latency_ms" in df_metrics["metric_name"].values:
            df_lat = df_metrics[df_metrics["metric_name"] == "latency_ms"]
            fig_lat = px.line(
                df_lat,
                x="timestamp" if "timestamp" in df_lat.columns else df_lat.index,
                y="metric_value",
                color="service",
                markers=True,
                title="Latency Timeseries by Service",
                template="plotly_dark",
            )
            fig_lat.update_layout(
                paper_bgcolor="#0d121f",
                plot_bgcolor="#0d121f",
                height=320,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_lat, use_container_width=True)
        else:
            st.info("No latency timeseries found in dataset.")

    with chart_col2:
        st.markdown("#### Error Rate (0.0 to 1.0)")
        if df_metrics is not None and not df_metrics.empty and "error_rate" in df_metrics["metric_name"].values:
            df_err = df_metrics[df_metrics["metric_name"] == "error_rate"]
            fig_err = px.line(
                df_err,
                x="timestamp" if "timestamp" in df_err.columns else df_err.index,
                y="metric_value",
                color="service",
                markers=True,
                title="Error Rate by Service",
                template="plotly_dark",
            )
            fig_err.update_layout(
                paper_bgcolor="#0d121f",
                plot_bgcolor="#0d121f",
                height=320,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_err, use_container_width=True)
        else:
            st.info("No error rate timeseries found in dataset.")

    st.markdown("#### CPU Utilization (%)")
    if df_metrics is not None and not df_metrics.empty and "cpu_percent" in df_metrics["metric_name"].values:
        df_cpu = df_metrics[df_metrics["metric_name"] == "cpu_percent"]
        fig_cpu = px.bar(
            df_cpu,
            x="service",
            y="metric_value",
            color="service",
            title="CPU Utilization by Service",
            template="plotly_dark",
        )
        fig_cpu.update_layout(
            paper_bgcolor="#0d121f",
            plot_bgcolor="#0d121f",
            height=280,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_cpu, use_container_width=True)

# TAB 4: EVIDENCE EXPLORER & RANKING TABLE
with tab_evidence:
    st.markdown("### 🔎 Ranked Root-Cause Candidates & Evidence Bundles")

    # Table of all ranked candidates
    table_rows = []
    for c in result.candidates:
        table_rows.append({
            "Rank": f"#{c.rank}",
            "Service": c.service,
            "Root-Cause Score": f"{c.score:.4f}",
            "Confidence": c.evidence_quality,
            "Metric Sig (A)": f"{c.metric_signal:.2f}",
            "Dependency (D)": f"{c.dependency_signal:.2f}",
            "Trace (T)": f"{c.trace_signal:.2f}",
            "Log (L)": f"{c.log_signal:.2f}",
            "Evidence Count": len(c.supporting_evidence_ids),
            "Reason": c.concise_reason,
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### Inspect Candidate Evidence Bundle")
    inspected_svc = st.selectbox(
        "Select Service to Inspect Grounded Telemetry",
        [c.service for c in result.candidates],
        index=0 if result.candidates else None,
    )

    if inspected_svc and inspected_svc in result.bundles:
        bundle = result.bundles[inspected_svc]

        ev_tab1, ev_tab2, ev_tab3, ev_tab4 = st.tabs(["Metrics Evidence", "Logs Evidence", "Trace Spans", "Audit Verification"])

        with ev_tab1:
            st.markdown("##### Verified Metrics Telemetry")
            if bundle.metrics_summary:
                st.dataframe(pd.DataFrame(bundle.metrics_summary), use_container_width=True, hide_index=True)
            else:
                st.write("No metrics records for this service.")

        with ev_tab2:
            st.markdown("##### Error & System Logs")
            if bundle.logs_summary:
                st.dataframe(pd.DataFrame(bundle.logs_summary), use_container_width=True, hide_index=True)
            else:
                st.write("No log entries recorded for this service.")

        with ev_tab3:
            st.markdown("##### Distributed Traces (Latency & Spans)")
            if bundle.traces_summary:
                st.dataframe(pd.DataFrame(bundle.traces_summary), use_container_width=True, hide_index=True)
            else:
                st.write("No trace spans recorded for this service.")

        with ev_tab4:
            st.markdown("##### Cryptographic-Style Citation Audit")
            st.markdown(f"**Verification Status:** `{result.verification_result.audit_status}`")
            for chk in result.verification_result.checks:
                st.markdown(f"- ✅ {chk}")
            if result.verification_result.failure_log:
                st.markdown("**Verification Failures:**")
                st.json(result.verification_result.failure_log)

# TAB 5: EVALUATION & BENCHMARK
with tab_benchmark:
    st.markdown("### 🏆 Algorithm Evaluation & Benchmarking Suite")
    st.caption("Evaluates deterministic ranking algorithms against labelled incidents with ground-truth separation.")

    if st.button("🚀 Run Full Benchmark Evaluation Across All Scenarios", use_container_width=True):
        evaluator = BenchmarkRunner(pipeline)
        test_cases = []
        for inc in incidents_meta:
            inc_id = inc["incident_id"]
            m, l, t, d = loader.load_incident_telemetry(inc_id)
            test_cases.append({
                "incident_id": inc_id,
                "ground_truth": inc["ground_truth_root_cause"],
                "metrics": m,
                "logs": l,
                "traces": t,
                "dependencies": d,
            })

        summary_base = evaluator.evaluate_suite(test_cases, method="baseline", dataset_name="RootSight Built-in Suite")
        summary_pr = evaluator.evaluate_suite(test_cases, method="pagerank", dataset_name="RootSight Built-in Suite")

        b_col1, b_col2, b_col3, b_col4 = st.columns(4)
        with b_col1:
            st.metric("Total Test Incidents", summary_base.total_incidents)
        with b_col2:
            st.metric("Baseline Top-1 Accuracy", f"{summary_base.top1_accuracy * 100:.1f}%")
        with b_col3:
            st.metric("Baseline Top-3 Accuracy", f"{summary_base.top3_accuracy * 100:.1f}%")
        with b_col4:
            st.metric("Baseline MRR", f"{summary_base.mrr:.3f}")

        pr_col1, pr_col2, pr_col3, pr_col4 = st.columns(4)
        with pr_col1:
            st.metric("Evaluated Suite", "PageRank Comparison")
        with pr_col2:
            st.metric("PageRank Top-1 Accuracy", f"{summary_pr.top1_accuracy * 100:.1f}%")
        with pr_col3:
            st.metric("PageRank Top-3 Accuracy", f"{summary_pr.top3_accuracy * 100:.1f}%")
        with pr_col4:
            st.metric("PageRank MRR", f"{summary_pr.mrr:.3f}")

        st.markdown("#### Detailed Incident-by-Incident Evaluation Table")
        eval_rows = []
        for rb, rp in zip(summary_base.results, summary_pr.results):
            eval_rows.append({
                "Incident ID": rb.incident_id,
                "Ground Truth": rb.ground_truth,
                "Baseline Top-1": rb.predicted_top1,
                "Baseline Correct?": "✅ Yes" if rb.is_top1_correct else "❌ No",
                "PageRank Top-1": rp.predicted_top1,
                "PageRank Correct?": "✅ Yes" if rp.is_top1_correct else "❌ No",
            })
        st.dataframe(pd.DataFrame(eval_rows), use_container_width=True, hide_index=True)

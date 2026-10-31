"""
RootSight - FastAPI Backend Service
Exposes REST endpoints for telemetry ingestion, deterministic root-cause diagnosis,
graph inspection, report verification, and benchmark evaluation.
"""

from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import pandas as pd
from engine.storage import save_dataset, load_dataset, save_analysis, list_analyses

from engine.data_loader import DataLoader
from engine.pipeline import DiagnosisPipeline, PipelineResult
from evaluation.benchmark import BenchmarkRunner

app = FastAPI(
    title="RootSight API",
    description="Deterministic, Auditable Cloud Outage Root-Cause Analysis API",
    version="1.0.0",
)

loader = DataLoader()
pipeline = DiagnosisPipeline()
evaluator = BenchmarkRunner(pipeline)


class AnalyzeRequest(BaseModel):
    incident_id: Optional[str] = None
    ranking_method: Optional[str] = "baseline"  # 'baseline' or 'pagerank'
    metrics: Optional[List[Dict[str, Any]]] = None
    logs: Optional[List[Dict[str, Any]]] = None
    traces: Optional[List[Dict[str, Any]]] = None
    dependencies: Optional[List[Dict[str, Any]]] = None
    prefer_ai_narration: Optional[bool] = False


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "RootSight Root-Cause Analysis Engine",
        "version": "1.0.0",
    }


@app.get("/api/incidents")
def list_incidents():
    """Lists all built-in demonstration incidents and their metadata."""
    incidents = loader.load_incidents_metadata()
    return {"total": len(incidents), "incidents": incidents}


@app.get("/api/incidents/{incident_id}/results")
def get_incident_results(
    incident_id: str,
    ranking_method: str = "baseline",
):
    """
    Runs deterministic RCA pipeline on a built-in incident and returns structured diagnosis.
    """
    df_metrics, df_logs, df_traces, df_deps = loader.load_incident_telemetry(incident_id)
    if df_metrics.empty and df_logs.empty and df_traces.empty:
        raise HTTPException(
            status_code=404, detail=f"Incident '{incident_id}' not found or telemetry empty."
        )

    result = pipeline.run(
        incident_id=incident_id,
        df_metrics=df_metrics,
        df_logs=df_logs,
        df_traces=df_traces,
        df_dependencies=df_deps,
        ranking_method=ranking_method,
    )

    return {
        "incident_id": result.incident_id,
        "method_used": result.method_used,
        "summary": result.execution_summary,
        "top_candidate": result.top_candidate.to_dict() if result.top_candidate else None,
        "candidates": [c.to_dict() for c in result.candidates],
        "verification": result.verification_result.to_dict(),
        "report": result.report.to_dict(),
        "graph": result.graph_result.to_serializable(),
    }


@app.post("/api/analyze")
def analyze_telemetry(request: AnalyzeRequest):
    """
    Analyzes telemetry payloads supplied in request body or loads a built-in incident if ID is given.
    """
    if request.metrics or request.logs or request.traces or request.dependencies:
        df_m = pd.DataFrame(request.metrics) if request.metrics else pd.DataFrame()
        df_l = pd.DataFrame(request.logs) if request.logs else pd.DataFrame()
        df_t = pd.DataFrame(request.traces) if request.traces else pd.DataFrame()
        df_d = pd.DataFrame(request.dependencies) if request.dependencies else pd.DataFrame()
        for kind, frame, validator in (("metrics", df_m, loader.validate_metrics_df), ("logs", df_l, loader.validate_logs_df), ("traces", df_t, loader.validate_traces_df)):
            if not frame.empty:
                checked = validator(frame)
                if not checked.is_valid:
                    raise HTTPException(status_code=422, detail={"dataset": kind, "errors": checked.errors})
                if kind == "metrics": df_m = checked.cleaned_df
                elif kind == "logs": df_l = checked.cleaned_df
                else: df_t = checked.cleaned_df
        if not df_d.empty:
            checked = loader.validate_dependencies_df(df_d)
            if not checked.is_valid:
                raise HTTPException(status_code=422, detail={"dataset": "dependencies", "errors": checked.errors})
            df_d = checked.cleaned_df
    else:
        incident_id = request.incident_id or "INC-001-DB-SPIKE"
        saved = load_dataset(incident_id)
        if any(saved.values()):
            df_m, df_l, df_t, df_d = (pd.DataFrame(saved[k]) for k in ("metrics", "logs", "traces", "dependencies"))
        else:
            df_m, df_l, df_t, df_d = loader.load_incident_telemetry(incident_id)

    analysis_id = request.incident_id or ("custom-analysis" if request.metrics or request.logs or request.traces or request.dependencies else "INC-001-DB-SPIKE")
    result = pipeline.run(
        incident_id=analysis_id,
        df_metrics=df_m,
        df_logs=df_l,
        df_traces=df_t,
        df_dependencies=df_d,
        ranking_method=request.ranking_method or "baseline",
        prefer_ai_narration=request.prefer_ai_narration or False,
    )

    response = {
        "incident_id": result.incident_id,
        "method_used": result.method_used,
        "top_candidate": result.top_candidate.to_dict() if result.top_candidate else None,
        "candidates": [c.to_dict() for c in result.candidates],
        "verification": result.verification_result.to_dict(),
        "report": result.report.to_dict(),
        "graph": result.graph_result.to_serializable(),
    }
    save_analysis(result.incident_id, response)
    return response


@app.post("/api/ingest")
def ingest_telemetry(payload: AnalyzeRequest):
    """Validate, normalize, and persist telemetry rows in the local SQLite store."""
    datasets = {"metrics": payload.metrics or [], "logs": payload.logs or [], "traces": payload.traces or [], "dependencies": payload.dependencies or []}
    if not any(datasets.values()):
        raise HTTPException(status_code=422, detail="Supply at least one non-empty telemetry dataset.")
    clean = {}
    warnings = {}
    for kind, rows in datasets.items():
        if not rows:
            clean[kind] = []
            continue
        frame = pd.DataFrame(rows)
        validator = {"metrics": loader.validate_metrics_df, "logs": loader.validate_logs_df, "traces": loader.validate_traces_df, "dependencies": loader.validate_dependencies_df}.get(kind)
        if validator:
            report = validator(frame)
            if not report.is_valid:
                raise HTTPException(status_code=422, detail={"dataset": kind, "errors": report.errors})
            warnings[kind] = report.warnings
            clean[kind] = report.cleaned_df.astype(object).where(pd.notna(report.cleaned_df), None).to_dict(orient="records")
        else:
            clean[kind] = frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")
    incident_id = payload.incident_id or "uploaded-telemetry"
    rows_saved = save_dataset(incident_id, clean)
    return {"incident_id": incident_id, "rows_saved": rows_saved, "datasets": {k: len(v) for k, v in clean.items()}, "warnings": warnings, "source": "uploaded"}


@app.get("/api/telemetry/{incident_id}")
def retrieve_telemetry(incident_id: str):
    datasets = load_dataset(incident_id)
    if not any(datasets.values()):
        raise HTTPException(status_code=404, detail="No persisted uploaded telemetry found for this incident.")
    return {"incident_id": incident_id, "source": "uploaded", "datasets": datasets}


@app.delete("/api/telemetry/{incident_id}")
def reset_telemetry(incident_id: str):
    """Remove persisted uploaded telemetry for one incident ID."""
    from engine.storage import delete_incident
    deleted_batches = delete_incident(incident_id)
    return {"incident_id": incident_id, "deleted_batches": deleted_batches}


@app.get("/api/history")
def analysis_history(limit: int = Query(50, ge=1, le=200)):
    return {"analyses": list_analyses(limit)}


@app.get("/api/benchmark")
def get_benchmark():
    """
    Executes automated evaluation across all built-in incidents and compares
    Method A (Explainable Baseline) with Method B (Personalized PageRank).
    """
    metadata_list = loader.load_incidents_metadata()
    test_cases = []
    for inc in metadata_list:
        inc_id = inc["incident_id"]
        df_m, df_l, df_t, df_d = loader.load_incident_telemetry(inc_id)
        test_cases.append({
            "incident_id": inc_id,
            "ground_truth": inc["ground_truth_root_cause"],
            "metrics": df_m,
            "logs": df_l,
            "traces": df_t,
            "dependencies": df_d,
        })

    summary_baseline = evaluator.evaluate_suite(test_cases, method="baseline")
    summary_pagerank = evaluator.evaluate_suite(test_cases, method="pagerank")

    return {
        "dataset": "RootSight Demonstration Ground-Truth Suite",
        "total_test_cases": len(test_cases),
        "baseline_evaluation": summary_baseline.to_dict(),
        "pagerank_evaluation": summary_pagerank.to_dict(),
    }

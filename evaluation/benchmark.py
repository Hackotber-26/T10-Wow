"""
RootSight - Benchmark and Evaluation Suite
Executes batch evaluation of RCA algorithms against ground-truth labelled incidents.
Calculates Top-1, Top-3, and MRR metrics for both Baseline and Personalized PageRank.
Provides adapter for RCAEval benchmark dataset schema when available.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import json

from engine.pipeline import DiagnosisPipeline, PipelineResult
from evaluation.metrics import IncidentEvaluationResult, BenchmarkSummary, compute_benchmark_metrics


class BenchmarkRunner:
    """
    Evaluates RootSight algorithms across a collection of incidents with ground truth.
    Ground-truth labels are kept strictly separate from the diagnosis pipeline input.
    """

    def __init__(self, pipeline: Optional[DiagnosisPipeline] = None):
        self.pipeline = pipeline or DiagnosisPipeline()

    def evaluate_incident(
        self,
        incident_id: str,
        ground_truth_root_cause: str,
        df_metrics: Optional[pd.DataFrame],
        df_logs: Optional[pd.DataFrame],
        df_traces: Optional[pd.DataFrame],
        df_dependencies: Optional[pd.DataFrame] = None,
        method: str = "baseline",
    ) -> IncidentEvaluationResult:
        """
        Runs the diagnosis pipeline on one incident and evaluates prediction against ground truth.
        """
        # Run diagnosis WITHOUT passing ground truth
        result = self.pipeline.run(
            incident_id=incident_id,
            df_metrics=df_metrics,
            df_logs=df_logs,
            df_traces=df_traces,
            df_dependencies=df_dependencies,
            ranking_method=method,
        )

        candidates = result.candidates
        predicted_top1 = candidates[0].service if candidates else "UNKNOWN"
        predicted_top3 = [c.service for c in candidates[:3]]

        gt_norm = str(ground_truth_root_cause).strip().lower()

        # Handle healthy or unknown ground truth cases
        if gt_norm in ["none", "healthy", "normal", "unknown"]:
            is_top1 = (predicted_top1.lower() in ["none", "unknown"] or result.top_candidate.score < 0.15)
            is_top3 = is_top1
            rank_of_truth = 1 if is_top1 else None
            reciprocal_rank = 1.0 if is_top1 else 0.0
        else:
            # Standard root cause evaluation
            rank_of_truth = None
            for c in candidates:
                if c.service.strip().lower() == gt_norm:
                    rank_of_truth = c.rank
                    break

            is_top1 = (predicted_top1.strip().lower() == gt_norm)
            is_top3 = any(p.strip().lower() == gt_norm for p in predicted_top3)
            reciprocal_rank = (1.0 / rank_of_truth) if rank_of_truth else 0.0

        top_qual = candidates[0].evidence_quality if candidates else "INSUFFICIENT_EVIDENCE"

        return IncidentEvaluationResult(
            incident_id=incident_id,
            ground_truth=ground_truth_root_cause,
            predicted_top1=predicted_top1,
            predicted_top3=predicted_top3,
            is_top1_correct=is_top1,
            is_top3_correct=is_top3,
            rank_of_truth=rank_of_truth,
            reciprocal_rank=reciprocal_rank,
            confidence_level=top_qual,
            method=result.method_used,
            notes=f"Top-1 Score: {result.top_candidate.score if result.top_candidate else 0.0:.2f}",
        )

    def evaluate_suite(
        self,
        incidents_data: List[Dict[str, Any]],
        method: str = "baseline",
        dataset_name: str = "RootSight Built-in Suite",
    ) -> BenchmarkSummary:
        """
        Runs evaluation over a suite of incidents.
        Each incident_data dict contains:
          - 'incident_id'
          - 'ground_truth'
          - 'metrics' (DataFrame)
          - 'logs' (DataFrame)
          - 'traces' (DataFrame)
          - 'dependencies' (optional DataFrame)
        """
        eval_results: List[IncidentEvaluationResult] = []
        for inc in incidents_data:
            res = self.evaluate_incident(
                incident_id=inc["incident_id"],
                ground_truth_root_cause=inc["ground_truth"],
                df_metrics=inc.get("metrics"),
                df_logs=inc.get("logs"),
                df_traces=inc.get("traces"),
                df_dependencies=inc.get("dependencies"),
                method=method,
            )
            eval_results.append(res)

        return compute_benchmark_metrics(
            eval_results,
            method_name="Explainable Baseline" if method == "baseline" else "Personalized PageRank",
            dataset_name=dataset_name,
            total_expected=len(incidents_data),
        )

    @staticmethod
    def load_rcaeval_incident(
        metadata: Dict[str, Any],
        metrics_df: pd.DataFrame,
        traces_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Adapter for RCAEval benchmark dataset schema.
        Maps RCAEval schema columns (cmdb_id, timestamp, metric_name, value) into RootSight format.
        """
        df_m = metrics_df.copy()
        if "cmdb_id" in df_m.columns and "service" not in df_m.columns:
            df_m["service"] = df_m["cmdb_id"]
        if "kpi_name" in df_m.columns and "metric_name" not in df_m.columns:
            df_m["metric_name"] = df_m["kpi_name"]
        if "value" in df_m.columns and "metric_value" not in df_m.columns:
            df_m["metric_value"] = df_m["value"]

        return {
            "incident_id": metadata.get("incident_id", "rcaeval-incident"),
            "ground_truth": metadata.get("root_cause", metadata.get("fault_service", "unknown")),
            "metrics": df_m,
            "logs": pd.DataFrame(),
            "traces": traces_df if traces_df is not None else pd.DataFrame(),
            "dependencies": pd.DataFrame(),
        }

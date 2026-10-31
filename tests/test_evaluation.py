"""
Unit tests for Benchmarking and Evaluation module.
Covers: Top-1 accuracy, Top-3 accuracy, MRR calculation,
strict denominator preservation, and RCAEval adapter schema conversion.
"""

import pytest
import pandas as pd
from evaluation.metrics import IncidentEvaluationResult, compute_benchmark_metrics
from evaluation.benchmark import BenchmarkRunner


def test_top1_and_top3_accuracy_calculation():
    results = [
        # Case 1: Top-1 Correct
        IncidentEvaluationResult(
            incident_id="I1", ground_truth="db", predicted_top1="db", predicted_top3=["db", "api", "cache"],
            is_top1_correct=True, is_top3_correct=True, rank_of_truth=1, reciprocal_rank=1.0,
            confidence_level="HIGH_CONFIDENCE", method="Baseline",
        ),
        # Case 2: Top-3 Correct (Rank 2)
        IncidentEvaluationResult(
            incident_id="I2", ground_truth="auth", predicted_top1="gw", predicted_top3=["gw", "auth", "cache"],
            is_top1_correct=False, is_top3_correct=True, rank_of_truth=2, reciprocal_rank=0.5,
            confidence_level="MODERATE_CONFIDENCE", method="Baseline",
        ),
        # Case 3: Completely Missed (Rank > 3)
        IncidentEvaluationResult(
            incident_id="I3", ground_truth="queue", predicted_top1="gw", predicted_top3=["gw", "auth", "cache"],
            is_top1_correct=False, is_top3_correct=False, rank_of_truth=5, reciprocal_rank=0.2,
            confidence_level="LOW_CONFIDENCE", method="Baseline",
        ),
        # Case 4: Absent from candidates (rank_of_truth = None)
        IncidentEvaluationResult(
            incident_id="I4", ground_truth="billing", predicted_top1="gw", predicted_top3=["gw", "auth", "cache"],
            is_top1_correct=False, is_top3_correct=False, rank_of_truth=None, reciprocal_rank=0.0,
            confidence_level="INSUFFICIENT_EVIDENCE", method="Baseline",
        ),
    ]

    summary = compute_benchmark_metrics(results, method_name="Baseline", dataset_name="Test Suite", total_expected=4)

    assert summary.total_incidents == 4
    assert summary.evaluated_incidents == 4
    assert summary.top1_correct_count == 1
    assert summary.top3_correct_count == 2
    # 1/4 = 0.25
    assert summary.top1_accuracy == pytest.approx(0.25)
    # 2/4 = 0.50
    assert summary.top3_accuracy == pytest.approx(0.50)
    # MRR = (1.0 + 0.5 + 0.2 + 0.0) / 4 = 1.7 / 4 = 0.425
    assert summary.mrr == pytest.approx(0.425)


def test_denominator_not_reduced_for_missing_cases():
    """Unranked or missing cases must remain in the denominator."""
    results = [
        IncidentEvaluationResult(
            incident_id="I1", ground_truth="db", predicted_top1="db", predicted_top3=["db"],
            is_top1_correct=True, is_top3_correct=True, rank_of_truth=1, reciprocal_rank=1.0,
            confidence_level="HIGH_CONFIDENCE", method="Baseline",
        )
    ]
    # Expected 5 cases, but only 1 succeeded
    summary = compute_benchmark_metrics(results, method_name="Baseline", dataset_name="Test", total_expected=5)
    assert summary.total_incidents == 5
    assert summary.top1_accuracy == pytest.approx(0.20)  # 1 / 5 = 0.20


def test_rcaeval_schema_adapter():
    rca_metrics = pd.DataFrame([
        {"cmdb_id": "service-node-1", "kpi_name": "cpu_util", "value": 95.5, "timestamp": 1600000000}
    ])
    meta = {"incident_id": "rcaeval-01", "root_cause": "service-node-1"}

    adapted = BenchmarkRunner.load_rcaeval_incident(meta, rca_metrics)
    assert adapted["incident_id"] == "rcaeval-01"
    assert adapted["ground_truth"] == "service-node-1"
    assert "service" in adapted["metrics"].columns
    assert "metric_name" in adapted["metrics"].columns
    assert "metric_value" in adapted["metrics"].columns
    assert adapted["metrics"]["service"].iloc[0] == "service-node-1"

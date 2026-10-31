"""
Unit tests for EvidenceCollector.
Covers: Evidence bundle assembly, separation of facts from inferences,
grounding of citations, and recommended investigation steps.
"""

import pytest
import pandas as pd
from engine.graph_builder import DependencyGraphBuilder
from engine.ranking import DeterministicRankingEngine, CandidateRanking
from detector.anomaly import AnomalyRecord
from reporting.evidence import EvidenceCollector


def test_evidence_bundle_separation_of_facts_and_inferences():
    collector = EvidenceCollector()
    builder = DependencyGraphBuilder()

    df_deps = pd.DataFrame([{"source_service": "api-gw", "destination_service": "db"}])
    graph_res = builder.build_graph(df_dependencies=df_deps)

    df_metrics = pd.DataFrame([
        {"service": "db", "metric_name": "latency_ms", "metric_value": 980.0, "timestamp": "2026-10-30T14:10:00Z"}
    ])
    df_logs = pd.DataFrame([
        {"service": "db", "level": "ERROR", "message": "Lock timeout", "log_id": "LOG-DB-1", "timestamp": "2026-10-30T14:10:01Z"}
    ])
    df_traces = pd.DataFrame([
        {"trace_id": "TR-10", "source_service": "api-gw", "destination_service": "db", "latency_ms": 990.0, "timestamp": "2026-10-30T14:10:00Z"}
    ])

    anom = AnomalyRecord(
        service="db",
        metric="latency_ms",
        observed_value=980.0,
        baseline_value=15.0,
        anomaly_score=0.92,
        severity="CRITICAL",
        detection_method="robust_z_mad",
        evidence_reference="METRIC-db-latency_ms-0",
    )

    candidate = CandidateRanking(
        service="db",
        rank=1,
        score=0.88,
        ranking_method="Baseline",
        metric_signal=0.92,
        dependency_signal=0.6,
        trace_signal=0.85,
        log_signal=0.9,
        supporting_evidence_ids=["METRIC-db-latency_ms-0"],
        concise_reason="High latency",
        evidence_quality="HIGH_CONFIDENCE",
    )

    bundle = collector.build_candidate_bundle(
        candidate=candidate,
        df_metrics=df_metrics,
        df_logs=df_logs,
        df_traces=df_traces,
        anomalies=[anom],
        graph_result=graph_res,
    )

    assert bundle.service == "db"
    assert len(bundle.observed_facts) > 0
    assert len(bundle.algorithmic_inferences) > 0
    assert len(bundle.recommended_actions) > 0
    assert "METRIC-db-latency_ms-0" in bundle.cited_evidence_ids
    assert "LOG-DB-1" in bundle.cited_evidence_ids
    assert "TR-10" in bundle.cited_evidence_ids

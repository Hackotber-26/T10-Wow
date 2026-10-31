"""
Unit tests for EvidenceVerifier.
Covers: Telemetry citation verification, rejection of phantom IDs,
unsupported claims detection, and audit status reporting.
"""

import pytest
import pandas as pd
from reporting.evidence import EvidenceBundle
from reporting.verifier import EvidenceVerifier


def test_valid_evidence_passes_verification():
    verifier = EvidenceVerifier()

    df_metrics = pd.DataFrame([
        {"service": "postgres-db", "metric_name": "latency_ms", "metric_value": 950.0}
    ])
    df_logs = pd.DataFrame([
        {"service": "postgres-db", "level": "ERROR", "message": "lock error", "log_id": "LOG-PG-1"}
    ])
    df_traces = pd.DataFrame([
        {"trace_id": "TR-001", "source_service": "cart", "destination_service": "postgres-db", "latency_ms": 950.0}
    ])

    bundle = EvidenceBundle(
        service="postgres-db",
        rank=1,
        score=0.9,
        ranking_method="Baseline",
        observed_facts=[
            {"evidence_id": "METRIC-postgres-db-latency_ms-0", "type": "METRIC", "description": "Latency 950ms"}
        ],
        algorithmic_inferences=[],
        recommended_actions=["Check locks"],
        cited_evidence_ids=["METRIC-postgres-db-latency_ms-0", "LOG-PG-1", "TR-001"],
        metrics_summary=[],
        logs_summary=[],
        traces_summary=[],
        dependencies_summary={},
    )

    res = verifier.verify_bundle(bundle, df_metrics, df_logs, df_traces)
    assert res.passed is True
    assert res.audit_status == "VERIFIED"
    assert res.failed_count == 0


def test_phantom_evidence_ids_rejected():
    """Verifier must catch and reject evidence IDs that do not exist in raw input telemetry."""
    verifier = EvidenceVerifier()

    df_metrics = pd.DataFrame([{"service": "postgres-db", "metric_name": "latency_ms", "metric_value": 950.0}])
    df_logs = pd.DataFrame()
    df_traces = pd.DataFrame()

    bundle = EvidenceBundle(
        service="postgres-db",
        rank=1,
        score=0.9,
        ranking_method="Baseline",
        observed_facts=[{"type": "FACT", "description": "some fact"}],
        algorithmic_inferences=[],
        recommended_actions=[],
        # "FAKE-ID-9999" was never in the metrics, logs, or traces
        cited_evidence_ids=["FAKE-ID-9999", "GHOST-TRACE-000"],
        metrics_summary=[],
        logs_summary=[],
        traces_summary=[],
        dependencies_summary={},
    )

    res = verifier.verify_bundle(bundle, df_metrics, df_logs, df_traces)
    assert res.passed is False
    assert res.audit_status == "VERIFICATION_FAILED"
    assert res.failed_count == 2
    assert any("FAKE-ID-9999" in str(item) for item in res.failure_log)

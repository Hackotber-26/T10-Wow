"""
Integration tests for RootSight End-to-End Diagnosis Pipeline.
Tests all 6 demonstration scenarios, template fallback when AI is unavailable,
missing telemetry components, and empty inputs.
"""

import pytest
import pandas as pd
import networkx as nx
from engine.data_loader import DataLoader
from engine.pipeline import DiagnosisPipeline


@pytest.fixture
def loader():
    return DataLoader()


@pytest.fixture
def pipeline():
    return DiagnosisPipeline()


def test_scenario_database_latency_spike(loader, pipeline):
    m, l, t, d = loader.load_incident_telemetry("INC-001-DB-SPIKE")
    res = pipeline.run("INC-001-DB-SPIKE", m, l, t, d, ranking_method="baseline")
    assert res.top_candidate is not None
    assert res.top_candidate.service == "postgres-db"
    assert res.top_candidate.score >= 0.70
    assert res.verification_result.passed is True


def test_ecommerce_demo_cascade_has_ordered_evidence_and_inventory_impact(loader, pipeline):
    metrics, logs, traces, dependencies = loader.load_incident_telemetry("INC-001-DB-SPIKE")
    result = pipeline.run("INC-001-DB-SPIKE", metrics, logs, traces, dependencies)
    pg = metrics[(metrics.service == "postgres-db") & (metrics.metric_name == "latency_ms")]
    assert pg[pg.timestamp < "2026-10-30T14:11:00Z"].metric_value.median() < 50
    assert pg[pg.timestamp >= "2026-10-30T14:11:00Z"].metric_value.median() > 800
    event_time = [logs[logs.service == service].timestamp.min() for service in ("postgres-db", "payment-service", "api-gateway", "inventory-service")]
    assert event_time == sorted(event_time)
    assert result.top_candidate.service == "postgres-db"
    assert "inventory-service" in result.graph_result.graph
    assert "inventory-service" in nx.ancestors(result.graph_result.graph, "postgres-db")


def test_scenario_payment_service_failure(loader, pipeline):
    m, l, t, d = loader.load_incident_telemetry("INC-002-PAY-FAIL")
    res = pipeline.run("INC-002-PAY-FAIL", m, l, t, d, ranking_method="baseline")
    assert res.top_candidate is not None
    assert res.top_candidate.service == "payment-service"
    assert res.top_candidate.score >= 0.70
    assert res.verification_result.passed is True


def test_scenario_api_latency_spike(loader, pipeline):
    m, l, t, d = loader.load_incident_telemetry("INC-003-API-GATEWAY")
    res = pipeline.run("INC-003-API-GATEWAY", m, l, t, d, ranking_method="baseline")
    assert res.top_candidate is not None
    assert res.top_candidate.service == "api-gateway"
    assert res.top_candidate.score >= 0.70


def test_scenario_inventory_service_overload(loader, pipeline):
    m, l, t, d = loader.load_incident_telemetry("INC-004-INV-OVERLOAD")
    res = pipeline.run("INC-004-INV-OVERLOAD", m, l, t, d, ranking_method="baseline")
    assert res.top_candidate is not None
    assert res.top_candidate.service == "inventory-service"
    assert res.top_candidate.score >= 0.70


def test_scenario_normal_healthy_system(loader, pipeline):
    m, l, t, d = loader.load_incident_telemetry("INC-005-HEALTHY")
    res = pipeline.run("INC-005-HEALTHY", m, l, t, d, ranking_method="baseline")
    # In healthy system, no anomalies exceed threshold, score remains below critical threshold
    assert len(res.anomalies) == 0
    assert res.top_candidate.score < 0.20


def test_scenario_insufficient_evidence(loader, pipeline):
    m, l, t, d = loader.load_incident_telemetry("INC-006-INSUFFICIENT")
    res = pipeline.run("INC-006-INSUFFICIENT", m, l, t, d, ranking_method="baseline")
    assert res.report.confidence_level in ["INSUFFICIENT_EVIDENCE", "LOW_CONFIDENCE"]


def test_ai_unavailable_fallback_to_template(loader, pipeline):
    """When prefer_ai_narration=True but no LLM API key exists, fallback is automatic and report is complete."""
    m, l, t, d = loader.load_incident_telemetry("INC-001-DB-SPIKE")
    res = pipeline.run("INC-001-DB-SPIKE", m, l, t, d, prefer_ai_narration=True)
    assert res.report.narrator_type == "DETERMINISTIC_TEMPLATE"
    assert len(res.report.summary) > 20
    assert len(res.report.suggested_next_steps) > 0


def test_pipeline_empty_data(pipeline):
    res = pipeline.run("EMPTY", pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    assert res.top_candidate is None
    assert res.report.confidence_level == "INSUFFICIENT_EVIDENCE"

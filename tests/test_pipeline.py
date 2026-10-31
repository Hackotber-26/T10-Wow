"""
Integration tests for RootSight End-to-End Diagnosis Pipeline.
Tests all 6 demonstration scenarios, template fallback when AI is unavailable,
missing telemetry components, and empty inputs.
"""

import pytest
import pandas as pd
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

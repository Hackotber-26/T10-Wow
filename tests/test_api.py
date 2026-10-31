import pytest
from fastapi import HTTPException

import api


@pytest.fixture
def isolated_api_db(tmp_path, monkeypatch):
    monkeypatch.setenv("ROOTSIGHT_DB", str(tmp_path / "rootsight.sqlite3"))


def test_health_catalog_builtin_results_and_benchmark(isolated_api_db):
    assert api.health_check()["status"] == "healthy"
    catalog = api.list_incidents()
    assert catalog["total"] >= 1
    incident_id = catalog["incidents"][0]["incident_id"]
    diagnosis = api.get_incident_results(incident_id)
    assert diagnosis["report"]["narrator_type"] == "DETERMINISTIC_TEMPLATE"
    benchmark = api.get_benchmark()
    assert benchmark["total_test_cases"] == catalog["total"]


def test_ingest_validation_analysis_retrieval_history_and_report(isolated_api_db):
    incident_id = "INC-API-E2E-TEST"
    metric_rows = [
        {"service": "postgres-db", "metric_name": "latency_ms", "metric_value": n, "timestamp": f"2026-10-31T00:0{i}:00Z"}
        for i, n in enumerate([14, 15, 14, 16, 15, 950])
    ]
    trace_rows = [
        {"trace_id": f"trace-{i}", "source_service": "checkout", "destination_service": "postgres-db", "latency_ms": n}
        for i, n in enumerate([14, 15, 14, 16, 15, 950])
    ]
    request = api.AnalyzeRequest(
        incident_id=incident_id,
        metrics=metric_rows,
        logs=[{"service": "checkout", "level": "ERROR", "message": "database timeout", "log_id": "log-1"}],
        traces=trace_rows,
        dependencies=[{"source_service": "checkout", "destination_service": "postgres-db", "call_count": 6}],
    )
    saved = api.ingest_telemetry(request)
    assert saved["rows_saved"] == 14
    assert len(api.retrieve_telemetry(incident_id)["datasets"]["metrics"]) == 6

    result = api.analyze_telemetry(api.AnalyzeRequest(incident_id=incident_id))
    assert result["candidates"]
    assert result["report"]["narrator_type"] == "DETERMINISTIC_TEMPLATE"
    assert result["graph"]["total_dependencies"] >= 1
    assert any(row["incident_id"] == incident_id for row in api.analysis_history(50)["analyses"])

    with pytest.raises(HTTPException) as error:
        api.ingest_telemetry(api.AnalyzeRequest(incident_id="bad", metrics=[{"service": "x"}]))
    assert error.value.status_code == 422
    assert api.reset_telemetry(incident_id)["deleted_batches"] == 4
    with pytest.raises(HTTPException) as missing:
        api.retrieve_telemetry(incident_id)
    assert missing.value.status_code == 404


def test_analyze_without_id_uses_synthetic_demo(isolated_api_db):
    response = api.analyze_telemetry(api.AnalyzeRequest())
    assert response["incident_id"] == "INC-001-DB-SPIKE"

from engine.data_loader import DataLoader
from engine.pipeline import DiagnosisPipeline


def test_offline_report_contains_ranked_findings_evidence_and_limitations(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    loader = DataLoader()
    metrics, logs, traces, dependencies = loader.load_incident_telemetry("INC-001-DB-SPIKE")
    result = DiagnosisPipeline().run(
        "INC-001-DB-SPIKE", metrics, logs, traces, dependencies,
        prefer_ai_narration=True,
    )
    report = result.report.to_dict()
    assert report["narrator_type"] == "DETERMINISTIC_TEMPLATE"
    assert report["leading_candidate"] == result.candidates[0].service
    assert report["ranking_rationale"]
    assert report["limitations_and_uncertainty"]
    assert result.bundles[result.candidates[0].service].cited_evidence_ids


def test_empty_report_explains_missing_evidence():
    result = DiagnosisPipeline().run("empty-case")
    assert result.report.leading_candidate in ("None", "None (Healthy or Inconclusive)")
    assert "without input telemetry" in result.report.limitations_and_uncertainty.lower()

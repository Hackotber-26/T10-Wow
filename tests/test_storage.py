from engine.storage import delete_incident, list_analyses, load_dataset, save_analysis, save_dataset


def test_telemetry_replace_retrieve_and_delete(tmp_path, monkeypatch):
    monkeypatch.setenv("ROOTSIGHT_DB", str(tmp_path / "rootsight.sqlite3"))
    first = [{"service": "db", "metric_value": 4.0}]
    assert save_dataset("incident-x", {"metrics": first}) == 1
    assert load_dataset("incident-x")["metrics"] == first
    assert save_dataset("incident-x", {"metrics": first}) == 1
    assert len(load_dataset("incident-x")["metrics"]) == 1
    assert delete_incident("incident-x") == 1
    assert not any(load_dataset("incident-x").values())


def test_analysis_history_is_persisted(tmp_path, monkeypatch):
    monkeypatch.setenv("ROOTSIGHT_DB", str(tmp_path / "rootsight.sqlite3"))
    saved_id = save_analysis("incident-y", {"top_candidate": "db"})
    history = list_analyses()
    assert history[0]["id"] == saved_id
    assert history[0]["result"]["top_candidate"] == "db"

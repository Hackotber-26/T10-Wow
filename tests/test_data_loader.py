import pandas as pd

from engine.data_loader import DataLoader


def test_dependency_validation_normalizes_drops_invalid_and_deduplicates():
    frame = pd.DataFrame([
        {"source": "api", "destination": "db", "call_count": 2},
        {"source": "api", "destination": "db", "call_count": 2},
        {"source": "", "destination": "cache", "call_count": 1},
        {"source": "api", "destination": "api", "call_count": 1},
        {"source": "api", "destination": "cache", "call_count": "bad"},
    ])
    result = DataLoader.validate_dependencies_df(frame)
    assert result.is_valid
    assert len(result.cleaned_df) == 1
    assert result.cleaned_df.iloc[0]["source_service"] == "api"
    assert result.cleaned_df.iloc[0]["destination_service"] == "db"
    assert len(result.warnings) == 3


def test_dependency_validation_rejects_missing_endpoints():
    result = DataLoader.validate_dependencies_df(pd.DataFrame([{"source": "api"}]))
    assert not result.is_valid
    assert "destination_service" in result.errors[0]

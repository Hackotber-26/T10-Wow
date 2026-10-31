"""
Unit tests for RootSight AnomalyDetector.
Covers: MAD calculation, MAD=0 edge case, threshold fallback, missing values, empty data,
metric anomalies, log error surges, and trace latency anomalies.
"""

import pytest
import numpy as np
import pandas as pd
from detector.anomaly import AnomalyDetector, AnomalyRecord


def test_mad_calculation_standard():
    detector = AnomalyDetector()
    data = np.array([10.0, 10.2, 10.1, 9.9, 10.0, 50.0])
    median, mad = detector.calculate_mad(data)
    assert median == pytest.approx(10.05, 0.1)
    assert mad >= 0.0


def test_mad_zero_handled_safely():
    """Test MAD=0 (all baseline observations identical) does not divide by zero."""
    detector = AnomalyDetector()
    data = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    median, mad = detector.calculate_mad(data)
    assert mad == 0.0

    # Normal value equal to median
    z_equal = detector.compute_robust_z(10.0, median, mad)
    assert z_equal == 0.0

    # Elevated value with zero MAD
    z_elevated = detector.compute_robust_z(50.0, median, mad)
    assert not np.isnan(z_elevated)
    assert not np.isinf(z_elevated)
    assert z_elevated > 0.0


def test_empty_dataframe_detection():
    detector = AnomalyDetector()
    anoms_metrics = detector.detect_metric_anomalies(pd.DataFrame())
    anoms_logs = detector.detect_log_anomalies(pd.DataFrame())
    anoms_traces = detector.detect_trace_anomalies(pd.DataFrame())
    assert anoms_metrics == []
    assert anoms_logs == []
    assert anoms_traces == []


def test_missing_values_and_malformed_input():
    detector = AnomalyDetector()
    df = pd.DataFrame([
        {"service": "auth", "metric_name": "latency_ms", "metric_value": "not_a_number"},
        {"service": None, "metric_name": "latency_ms", "metric_value": 45.0},
        {"service": "auth", "metric_name": None, "metric_value": 45.0},
        {"service": "auth", "metric_name": "latency_ms", "metric_value": np.nan},
    ])
    anoms = detector.detect_metric_anomalies(df)
    assert isinstance(anoms, list)


def test_threshold_fallback_for_sparse_history():
    """Tests threshold fallback when observations < min_history_observations."""
    detector = AnomalyDetector(min_history_observations=5)
    # Only 2 observations (too few for robust MAD statistics)
    df = pd.DataFrame([
        {"service": "payment", "metric_name": "latency_ms", "metric_value": 950.0},
        {"service": "payment", "metric_name": "latency_ms", "metric_value": 900.0},
    ])
    anoms = detector.detect_metric_anomalies(df)
    assert len(anoms) > 0
    assert any(a.detection_method == "threshold_fallback" for a in anoms)
    assert any(a.severity == "CRITICAL" for a in anoms)


def test_log_error_surges():
    detector = AnomalyDetector()
    df_logs = pd.DataFrame([
        {"service": "cart", "level": "INFO", "message": "Item added", "log_id": "L1"},
        {"service": "cart", "level": "ERROR", "message": "DB connection timeout", "log_id": "L2"},
        {"service": "cart", "level": "ERROR", "message": "Transaction aborted", "log_id": "L3"},
        {"service": "cart", "level": "FATAL", "message": "Worker thread crash", "log_id": "L4"},
    ])
    anoms = detector.detect_log_anomalies(df_logs)
    assert len(anoms) == 1
    assert anoms[0].service == "cart"
    assert anoms[0].metric == "log_error_count"
    assert anoms[0].observed_value == 3.0


def test_trace_latency_spike():
    detector = AnomalyDetector()
    df_traces = pd.DataFrame([
        {"trace_id": "T1", "source_service": "api", "destination_service": "db", "latency_ms": 15.0},
        {"trace_id": "T2", "source_service": "api", "destination_service": "db", "latency_ms": 16.0},
        {"trace_id": "T3", "source_service": "api", "destination_service": "db", "latency_ms": 14.0},
        {"trace_id": "T4", "source_service": "api", "destination_service": "db", "latency_ms": 15.0},
        {"trace_id": "T5", "source_service": "api", "destination_service": "db", "latency_ms": 980.0},
    ])
    anoms = detector.detect_trace_anomalies(df_traces)
    assert len(anoms) > 0
    assert any(a.service == "db" and a.observed_value == 980.0 for a in anoms)

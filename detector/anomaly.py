"""
RootSight - Anomaly Detection Module
Implements robust statistical anomaly detection using Median Absolute Deviation (MAD),
robust z-score, and transparent threshold-based fallback.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd


@dataclass
class AnomalyRecord:
    service: str
    metric: str
    observed_value: float
    baseline_value: float
    anomaly_score: float  # Normalized 0.0 to 1.0
    severity: str        # 'CRITICAL', 'WARNING', 'INFO'
    detection_method: str  # 'robust_z_mad', 'threshold_fallback'
    evidence_reference: str  # Unique pointer to underlying raw record
    timestamp: Optional[str] = None
    z_score: Optional[float] = None
    direction: str = "high"  # 'high' or 'low'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AnomalyDetector:
    """
    Detects anomalies in metrics, logs, and traces using robust statistics.
    Formula: robust_z = 0.6745 * (value - median) / MAD
    """

    # Minimal significant delta required before flagging a deviation as an anomaly
    # Prevents false alarms on low-variance baseline jitter (e.g. 14ms -> 19ms)
    MIN_SIGNIFICANT_DELTAS = {
        "latency_ms": 50.0,
        "cpu_percent": 15.0,
        "memory_percent": 10.0,
        "error_rate": 0.03,
        "request_count": 10.0,
    }
    DEFAULT_THRESHOLDS = {
        "latency_ms": {"high": 350.0, "critical": 800.0},
        "error_rate": {"high": 0.05, "critical": 0.20},
        "cpu_percent": {"high": 80.0, "critical": 92.0},
        "memory_percent": {"high": 85.0, "critical": 95.0},
        "request_count": {"low": 5.0, "critical_low": 1.0},
    }

    def __init__(
        self,
        mad_threshold: float = 3.0,
        min_history_observations: int = 4,
        custom_thresholds: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.mad_threshold = float(mad_threshold)
        self.min_history_observations = int(min_history_observations)
        self.thresholds = dict(self.DEFAULT_THRESHOLDS)
        if custom_thresholds:
            for k, v in custom_thresholds.items():
                self.thresholds[k] = v

    @staticmethod
    def calculate_mad(values: np.ndarray) -> tuple[float, float]:
        """
        Calculate median and Median Absolute Deviation (MAD).
        Returns (median, mad).
        """
        if len(values) == 0:
            return 0.0, 0.0
        median = float(np.median(values))
        deviations = np.abs(values - median)
        mad = float(np.median(deviations))
        return median, mad

    def compute_robust_z(self, value: float, median: float, mad: float) -> float:
        """
        Computes robust z-score: 0.6745 * (value - median) / MAD.
        Safely handles MAD == 0.
        """
        if mad > 1e-9:
            return 0.6745 * (value - median) / mad
        
        # MAD is 0 (all or most baseline points are identical)
        diff = value - median
        if abs(diff) < 1e-9:
            return 0.0
        
        # When MAD is 0, use a scaled difference based on standard baseline scale
        # If median is non-zero, scale by |median|; otherwise fallback to absolute diff
        scale = abs(median) * 0.1 if abs(median) > 1e-6 else 1.0
        return diff / scale

    def normalize_score(self, z_score: float) -> float:
        """
        Normalize unbounded robust z-score to [0.0, 1.0] using sigmoid-like scaling.
        A z-score equal to mad_threshold maps to ~0.70.
        """
        abs_z = abs(z_score)
        if abs_z == 0.0:
            return 0.0
        # Smooth saturating function
        norm = 1.0 - np.exp(-0.4 * (abs_z / max(1.0, self.mad_threshold)))
        return float(np.clip(norm, 0.0, 1.0))

    def classify_severity(self, anomaly_score: float, z_score: Optional[float] = None) -> str:
        """Classify severity into CRITICAL, WARNING, or INFO."""
        if anomaly_score >= 0.70 or (z_score is not None and abs(z_score) >= 5.0):
            return "CRITICAL"
        elif anomaly_score >= 0.40 or (z_score is not None and abs(z_score) >= self.mad_threshold):
            return "WARNING"
        return "INFO"

    def detect_metric_anomalies(self, df_metrics: pd.DataFrame) -> List[AnomalyRecord]:
        """
        Inspect metrics timeseries for each (service, metric_name).
        Uses robust z-score when sufficient historical observations exist,
        otherwise falls back to threshold-based detection.
        """
        anomalies: List[AnomalyRecord] = []
        if df_metrics is None or df_metrics.empty:
            return anomalies

        required_cols = {"service", "metric_name", "metric_value"}
        if not required_cols.issubset(df_metrics.columns):
            return anomalies

        # Ensure numeric values
        df = df_metrics.copy()
        df["metric_value"] = pd.to_numeric(df["metric_value"], errors="coerce")
        df = df.dropna(subset=["service", "metric_name", "metric_value"])

        grouped = df.groupby(["service", "metric_name"])

        for (service, metric_name), group in grouped:
            values = group["metric_value"].to_numpy(dtype=float)
            timestamps = group["timestamp"].tolist() if "timestamp" in group.columns else [None] * len(values)
            record_ids = group.index.tolist()

            n_obs = len(values)
            if n_obs == 0:
                continue

            # Check if we have enough observations for robust statistics
            if n_obs >= self.min_history_observations:
                # Use pre-incident historical window (first 50-60% of points) as baseline reference
                base_len = max(self.min_history_observations, int(n_obs * 0.5)) if n_obs >= 6 else n_obs
                baseline_vals = values[:base_len]
                median, mad = self.calculate_mad(baseline_vals)
                for idx, val, ts in zip(record_ids, values, timestamps):
                    z = self.compute_robust_z(val, median, mad)
                    
                    # Direction of anomaly: high latency/error/cpu, or abnormally low throughput
                    is_suspicious = False
                    direction = "high"
                    if metric_name == "request_count":
                        if z < -self.mad_threshold:
                            min_delta = self.MIN_SIGNIFICANT_DELTAS.get(metric_name, 5.0)
                            if abs(val - median) >= min_delta:
                                is_suspicious = True
                                direction = "low"
                    else:
                        if z > self.mad_threshold:
                            min_delta = self.MIN_SIGNIFICANT_DELTAS.get(metric_name, 10.0)
                            # Ensure deviation is practically significant and not trivial jitter
                            if abs(val - median) >= min_delta:
                                is_suspicious = True
                                direction = "high"

                    if is_suspicious:
                        score = self.normalize_score(z)
                        severity = self.classify_severity(score, z)
                        ev_id = f"METRIC-{service}-{metric_name}-{idx}"
                        anomalies.append(
                            AnomalyRecord(
                                service=str(service),
                                metric=str(metric_name),
                                observed_value=float(val),
                                baseline_value=float(median),
                                anomaly_score=score,
                                severity=severity,
                                detection_method="robust_z_mad",
                                evidence_reference=ev_id,
                                timestamp=str(ts) if ts is not None else None,
                                z_score=float(z),
                                direction=direction,
                            )
                        )
            else:
                # Fallback: threshold-based detection for sparse observations
                thresh_info = self.thresholds.get(metric_name, None)
                for idx, val, ts in zip(record_ids, values, timestamps):
                    is_anom = False
                    severity = "INFO"
                    direction = "high"
                    score = 0.0

                    if thresh_info:
                        if "critical" in thresh_info and val >= thresh_info["critical"]:
                            is_anom = True
                            severity = "CRITICAL"
                            score = 0.90
                        elif "high" in thresh_info and val >= thresh_info["high"]:
                            is_anom = True
                            severity = "WARNING"
                            score = 0.65
                        elif "critical_low" in thresh_info and val <= thresh_info["critical_low"]:
                            is_anom = True
                            severity = "CRITICAL"
                            score = 0.85
                            direction = "low"
                        elif "low" in thresh_info and val <= thresh_info["low"]:
                            is_anom = True
                            severity = "WARNING"
                            score = 0.50
                            direction = "low"

                    if is_anom:
                        baseline_val = thresh_info.get("high", 0.0) if thresh_info else 0.0
                        ev_id = f"METRIC-{service}-{metric_name}-{idx}"
                        anomalies.append(
                            AnomalyRecord(
                                service=str(service),
                                metric=str(metric_name),
                                observed_value=float(val),
                                baseline_value=float(baseline_val),
                                anomaly_score=score,
                                severity=severity,
                                detection_method="threshold_fallback",
                                evidence_reference=ev_id,
                                timestamp=str(ts) if ts is not None else None,
                                z_score=None,
                                direction=direction,
                            )
                        )

        return anomalies

    def detect_log_anomalies(self, df_logs: pd.DataFrame) -> List[AnomalyRecord]:
        """
        Detects surges in error-level logs per service.
        """
        anomalies: List[AnomalyRecord] = []
        if df_logs is None or df_logs.empty:
            return anomalies

        if "service" not in df_logs.columns or "level" not in df_logs.columns:
            return anomalies

        df = df_logs.copy()
        df["level_clean"] = df["level"].astype(str).str.upper().str.strip()

        # Group by service
        for service, group in df.groupby("service"):
            total_logs = len(group)
            error_logs = group[group["level_clean"].isin(["ERROR", "FATAL", "CRITICAL"])]
            error_count = len(error_logs)

            if error_count > 0:
                error_ratio = error_count / max(1, total_logs)
                score = min(1.0, 0.4 + (error_ratio * 0.6))
                severity = "CRITICAL" if error_count >= 5 or error_ratio >= 0.5 else "WARNING"
                first_err_id = error_logs.iloc[0].get("log_id", f"LOG-{service}-ERR-0")
                ts = error_logs.iloc[0].get("timestamp", None)

                anomalies.append(
                    AnomalyRecord(
                        service=str(service),
                        metric="log_error_count",
                        observed_value=float(error_count),
                        baseline_value=0.0,
                        anomaly_score=float(score),
                        severity=severity,
                        detection_method="log_error_frequency",
                        evidence_reference=str(first_err_id),
                        timestamp=str(ts) if ts is not None else None,
                        z_score=None,
                        direction="high",
                    )
                )

        return anomalies

    def detect_trace_anomalies(self, df_traces: pd.DataFrame) -> List[AnomalyRecord]:
        """
        Detects anomalous trace spans with extreme latencies.
        """
        anomalies: List[AnomalyRecord] = []
        if df_traces is None or df_traces.empty:
            return anomalies

        if "latency_ms" not in df_traces.columns:
            return anomalies

        df = df_traces.copy()
        df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")
        df = df.dropna(subset=["latency_ms"])

        target_col = "destination_service" if "destination_service" in df.columns else (
            "service" if "service" in df.columns else "source_service"
        )
        if target_col not in df.columns:
            return anomalies

        for target_svc, group in df.groupby(target_col):
            latencies = group["latency_ms"].to_numpy(dtype=float)
            if len(latencies) >= self.min_history_observations:
                median, mad = self.calculate_mad(latencies)
                for idx, row in group.iterrows():
                    val = float(row["latency_ms"])
                    z = self.compute_robust_z(val, median, mad)
                    if z > self.mad_threshold:
                        score = self.normalize_score(z)
                        severity = self.classify_severity(score, z)
                        trace_ref = str(row.get("trace_id", f"TRACE-{idx}"))
                        anomalies.append(
                            AnomalyRecord(
                                service=str(target_svc),
                                metric="trace_latency_ms",
                                observed_value=val,
                                baseline_value=float(median),
                                anomaly_score=score,
                                severity=severity,
                                detection_method="robust_z_mad",
                                evidence_reference=trace_ref,
                                timestamp=str(row.get("timestamp", "")) if "timestamp" in row else None,
                                z_score=float(z),
                                direction="high",
                            )
                        )
            else:
                # Threshold fallback (> 350ms is warning, > 800ms is critical)
                for idx, row in group.iterrows():
                    val = float(row["latency_ms"])
                    if val >= 350.0:
                        severity = "CRITICAL" if val >= 800.0 else "WARNING"
                        score = float(np.clip(0.50 + (val / 1200.0) * 0.50, 0.50, 1.0))
                        trace_ref = str(row.get("trace_id", f"TRACE-{idx}"))
                        anomalies.append(
                            AnomalyRecord(
                                service=str(target_svc),
                                metric="trace_latency_ms",
                                observed_value=val,
                                baseline_value=150.0,
                                anomaly_score=score,
                                severity=severity,
                                detection_method="threshold_fallback",
                                evidence_reference=trace_ref,
                                timestamp=str(row.get("timestamp", "")) if "timestamp" in row else None,
                                z_score=None,
                                direction="high",
                            )
                        )

        return anomalies

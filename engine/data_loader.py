"""
RootSight - Data Ingestion and Validation Layer
Loads metrics, logs, traces, and topologies from built-in scenarios or user uploads.
Validates required columns, coerces types, detects anomalies, and normalizes identifiers.
"""

import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd


@dataclass
class ValidationReport:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    row_count: int
    cleaned_df: Optional[pd.DataFrame] = None


class DataLoader:
    """
    Ingests, validates, and filters telemetry datasets.
    """

    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        self.data_dir = data_dir

    def load_incidents_metadata(self) -> List[Dict[str, Any]]:
        """Loads available incident scenarios from incidents.json."""
        incidents_path = os.path.join(self.data_dir, "incidents.json")
        if not os.path.exists(incidents_path):
            return []
        with open(incidents_path, "r") as f:
            return json.load(f)

    def load_incident_telemetry(
        self, incident_id: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads metrics, logs, traces, and dependencies for a specific incident ID.
        """
        metrics_p = os.path.join(self.data_dir, "metrics.csv")
        logs_p = os.path.join(self.data_dir, "logs.csv")
        traces_p = os.path.join(self.data_dir, "traces.csv")
        deps_p = os.path.join(self.data_dir, "dependencies.csv")

        df_metrics = pd.read_csv(metrics_p) if os.path.exists(metrics_p) else pd.DataFrame()
        df_logs = pd.read_csv(logs_p) if os.path.exists(logs_p) else pd.DataFrame()
        df_traces = pd.read_csv(traces_p) if os.path.exists(traces_p) else pd.DataFrame()
        df_deps = pd.read_csv(deps_p) if os.path.exists(deps_p) else pd.DataFrame()

        # Filter by incident_id if column exists
        if "incident_id" in df_metrics.columns:
            df_metrics = df_metrics[df_metrics["incident_id"] == incident_id].copy()
        if "incident_id" in df_logs.columns:
            df_logs = df_logs[df_logs["incident_id"] == incident_id].copy()
        if "incident_id" in df_traces.columns:
            df_traces = df_traces[df_traces["incident_id"] == incident_id].copy()
        if "incident_id" in df_deps.columns:
            df_deps = df_deps[df_deps["incident_id"] == incident_id].copy()

        # Normalize schemas
        df_metrics = self.normalize_metrics(df_metrics)
        df_logs = self.normalize_logs(df_logs)
        df_traces = self.normalize_traces(df_traces)

        return df_metrics, df_logs, df_traces, df_deps

    @staticmethod
    def validate_metrics_df(df: Optional[pd.DataFrame]) -> ValidationReport:
        errors = []
        warnings = []
        if df is None or df.empty:
            return ValidationReport(False, ["Metrics dataset is empty."], [], 0, None)

        required = {"service", "metric_name", "metric_value"}
        missing = required - set(df.columns)
        if missing:
            return ValidationReport(False, [f"Missing required columns: {missing}"], [], len(df), None)

        cleaned = df.copy()
        # Clean service names
        cleaned["service"] = cleaned["service"].astype(str).str.strip()
        cleaned["metric_name"] = cleaned["metric_name"].astype(str).str.strip()
        
        # Coerce numeric values
        initial_count = len(cleaned)
        cleaned["metric_value"] = pd.to_numeric(cleaned["metric_value"], errors="coerce")
        dropped_nan = cleaned["metric_value"].isna().sum()
        if dropped_nan > 0:
            warnings.append(f"Dropped {dropped_nan} row(s) with non-numeric metric_value.")
            cleaned = cleaned.dropna(subset=["metric_value"])

        if "timestamp" not in cleaned.columns:
            warnings.append("No 'timestamp' column provided in metrics; timestamps will be indexed.")

        return ValidationReport(True, errors, warnings, len(cleaned), cleaned)

    @staticmethod
    def validate_logs_df(df: Optional[pd.DataFrame]) -> ValidationReport:
        errors = []
        warnings = []
        if df is None or df.empty:
            return ValidationReport(True, [], ["Logs dataset is empty."], 0, pd.DataFrame())

        required = {"service", "level", "message"}
        missing = required - set(df.columns)
        if missing:
            return ValidationReport(False, [f"Missing required columns in logs: {missing}"], [], len(df), None)

        cleaned = df.copy()
        cleaned["service"] = cleaned["service"].astype(str).str.strip()
        cleaned["level"] = cleaned["level"].astype(str).str.upper().str.strip()
        cleaned["message"] = cleaned["message"].astype(str)

        if "log_id" not in cleaned.columns:
            warnings.append("Synthesizing log_id identifiers for records without log_id.")
            cleaned["log_id"] = [f"LOG-AUTO-{i}" for i in range(len(cleaned))]

        return ValidationReport(True, errors, warnings, len(cleaned), cleaned)

    @staticmethod
    def validate_traces_df(df: Optional[pd.DataFrame]) -> ValidationReport:
        errors = []
        warnings = []
        if df is None or df.empty:
            return ValidationReport(True, [], ["Traces dataset is empty."], 0, pd.DataFrame())

        # Check required columns (support both standard and simplified)
        has_src = "source_service" in df.columns or "source" in df.columns
        has_dst = "destination_service" in df.columns or "destination" in df.columns
        has_lat = "latency_ms" in df.columns

        if not (has_src and has_dst and has_lat):
            return ValidationReport(False, ["Traces requires source, destination, and latency_ms columns."], [], len(df), None)

        cleaned = df.copy()
        if "source" in cleaned.columns and "source_service" not in cleaned.columns:
            cleaned["source_service"] = cleaned["source"]
        if "destination" in cleaned.columns and "destination_service" not in cleaned.columns:
            cleaned["destination_service"] = cleaned["destination"]

        cleaned["source_service"] = cleaned["source_service"].astype(str).str.strip()
        cleaned["destination_service"] = cleaned["destination_service"].astype(str).str.strip()
        cleaned["latency_ms"] = pd.to_numeric(cleaned["latency_ms"], errors="coerce")
        cleaned = cleaned.dropna(subset=["latency_ms"])

        if "trace_id" not in cleaned.columns:
            warnings.append("Synthesizing trace_id identifiers.")
            cleaned["trace_id"] = [f"TR-AUTO-{i}" for i in range(len(cleaned))]

        return ValidationReport(True, errors, warnings, len(cleaned), cleaned)

    @classmethod
    def normalize_metrics(cls, df: pd.DataFrame) -> pd.DataFrame:
        rep = cls.validate_metrics_df(df)
        return rep.cleaned_df if rep.is_valid and rep.cleaned_df is not None else pd.DataFrame()

    @classmethod
    def normalize_logs(cls, df: pd.DataFrame) -> pd.DataFrame:
        rep = cls.validate_logs_df(df)
        return rep.cleaned_df if rep.is_valid and rep.cleaned_df is not None else pd.DataFrame()

    @classmethod
    def normalize_traces(cls, df: pd.DataFrame) -> pd.DataFrame:
        rep = cls.validate_traces_df(df)
        return rep.cleaned_df if rep.is_valid and rep.cleaned_df is not None else pd.DataFrame()

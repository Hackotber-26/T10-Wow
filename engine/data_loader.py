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
import numpy as np


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
        bad_values = cleaned["metric_value"].isna() | ~np.isfinite(cleaned["metric_value"])
        if bad_values.any():
            warnings.append(f"Dropped {int(bad_values.sum())} row(s) with invalid metric_value.")
            cleaned = cleaned.loc[~bad_values].copy()
        bad_identity = (cleaned["service"].str.lower().isin(["", "nan", "none"])) | (cleaned["metric_name"].str.lower().isin(["", "nan", "none"]))
        if bad_identity.any():
            warnings.append(f"Dropped {int(bad_identity.sum())} row(s) with blank service or metric_name.")
            cleaned = cleaned.loc[~bad_identity].copy()
        duplicates = int(cleaned.duplicated().sum())
        if duplicates:
            warnings.append(f"Removed {duplicates} exact duplicate metric row(s).")
            cleaned = cleaned.drop_duplicates().copy()

        if "timestamp" not in cleaned.columns:
            warnings.append("No 'timestamp' column provided in metrics; timestamps will be indexed.")

        return ValidationReport(bool(len(cleaned)), errors if len(cleaned) else ["No valid metric rows remain."], warnings, len(cleaned), cleaned)

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
        bad_identity = cleaned["service"].str.lower().isin(["", "nan", "none"])
        if bad_identity.any():
            warnings.append(f"Dropped {int(bad_identity.sum())} log row(s) with blank service.")
            cleaned = cleaned.loc[~bad_identity].copy()
        duplicates = int(cleaned.duplicated().sum())
        if duplicates:
            warnings.append(f"Removed {duplicates} exact duplicate log row(s).")
            cleaned = cleaned.drop_duplicates().copy()

        if "log_id" not in cleaned.columns:
            warnings.append("Synthesizing log_id identifiers for records without log_id.")
            cleaned["log_id"] = [f"LOG-AUTO-{i}" for i in range(len(cleaned))]

        return ValidationReport(bool(len(cleaned)), errors if len(cleaned) else ["No valid log rows remain."], warnings, len(cleaned), cleaned)

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
        bad_rows = cleaned["latency_ms"].isna() | ~np.isfinite(cleaned["latency_ms"]) | cleaned["source_service"].str.lower().isin(["", "nan", "none"]) | cleaned["destination_service"].str.lower().isin(["", "nan", "none"])
        if bad_rows.any():
            warnings.append(f"Dropped {int(bad_rows.sum())} trace row(s) with invalid endpoints or latency.")
            cleaned = cleaned.loc[~bad_rows].copy()
        duplicates = int(cleaned.duplicated().sum())
        if duplicates:
            warnings.append(f"Removed {duplicates} exact duplicate trace row(s).")
            cleaned = cleaned.drop_duplicates().copy()

        if "trace_id" not in cleaned.columns:
            warnings.append("Synthesizing trace_id identifiers.")
            cleaned["trace_id"] = [f"TR-AUTO-{i}" for i in range(len(cleaned))]

        return ValidationReport(bool(len(cleaned)), errors if len(cleaned) else ["No valid trace rows remain."], warnings, len(cleaned), cleaned)

    @staticmethod
    def validate_dependencies_df(df: Optional[pd.DataFrame]) -> ValidationReport:
        if df is None or df.empty:
            return ValidationReport(True, [], ["Dependencies dataset is empty."], 0, pd.DataFrame())
        cleaned = df.copy()
        if "source_service" not in cleaned and "source" in cleaned:
            cleaned["source_service"] = cleaned["source"]
        if "destination_service" not in cleaned and "destination" in cleaned:
            cleaned["destination_service"] = cleaned["destination"]
        required = {"source_service", "destination_service"}
        missing = required - set(cleaned.columns)
        if missing:
            return ValidationReport(False, [f"Dependencies missing columns: {sorted(missing)}"], [], len(cleaned), None)
        cleaned["source_service"] = cleaned["source_service"].fillna("").astype(str).str.strip()
        cleaned["destination_service"] = cleaned["destination_service"].fillna("").astype(str).str.strip()
        invalid = (cleaned["source_service"] == "") | (cleaned["destination_service"] == "") | (cleaned["source_service"] == cleaned["destination_service"])
        warnings = []
        if invalid.any():
            warnings.append(f"Dropped {int(invalid.sum())} dependency row(s) with blank or self-referential services.")
            cleaned = cleaned.loc[~invalid].copy()
        for col in ("weight", "call_count"):
            if col in cleaned.columns:
                cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
                bad = cleaned[col].isna() | ~np.isfinite(cleaned[col]) | (cleaned[col] < 0)
                if bad.any():
                    warnings.append(f"Dropped {int(bad.sum())} dependency row(s) with invalid {col}.")
                    cleaned = cleaned.loc[~bad].copy()
        duplicate_count = int(cleaned.duplicated().sum())
        if duplicate_count:
            warnings.append(f"Removed {duplicate_count} exact duplicate dependency row(s).")
            cleaned = cleaned.drop_duplicates().copy()
        return ValidationReport(bool(len(cleaned)), [] if len(cleaned) else ["No valid dependency rows remain."], warnings, len(cleaned), cleaned)

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

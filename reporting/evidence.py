"""
RootSight - Evidence Bundle Module
Extracts, structures, and categorizes verified telemetry evidence for each root-cause candidate.
Distinguishes between Observed Facts, Algorithmic Inferences, and Recommended Investigative Actions.
"""

from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional
import pandas as pd

from detector.anomaly import AnomalyRecord
from engine.ranking import CandidateRanking
from engine.graph_builder import GraphAnalysisResult


@dataclass
class EvidenceBundle:
    service: str
    rank: int
    score: float
    ranking_method: str
    observed_facts: List[Dict[str, Any]]
    algorithmic_inferences: List[Dict[str, Any]]
    recommended_actions: List[str]
    cited_evidence_ids: List[str]
    metrics_summary: List[Dict[str, Any]]
    logs_summary: List[Dict[str, Any]]
    traces_summary: List[Dict[str, Any]]
    dependencies_summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvidenceCollector:
    """
    Collects real telemetry evidence from raw datasets for a given candidate.
    """

    RECOMMENDED_ACTIONS_MAP = {
        "postgres-db": [
            "Inspect active transactions and lock contention via pg_stat_activity.",
            "Analyze slow query log for unindexed queries or long-running scans.",
            "Verify storage I/O throughput and disk volume saturation.",
        ],
        "payment-service": [
            "Check recent SSL/TLS certificate expiry and keystore configurations.",
            "Inspect payment gateway egress firewall rules and response status codes.",
            "Verify thread pool and connection pool saturation in payment-service runtime.",
        ],
        "api-gateway": [
            "Inspect CPU profile and garbage collection pause times on gateway pods.",
            "Verify rate-limiting and regex route matching performance.",
            "Check upstream DNS resolution latency and keepalive connections.",
        ],
        "inventory-service": [
            "Verify CPU and memory usage against resource limits and HPA configuration.",
            "Inspect cache hit ratio for inventory stock lookups.",
            "Check downstream database queue depth and connection pool availability.",
        ],
        "auth-service": [
            "Inspect token validation cache hit/miss ratio.",
            "Check Redis / database latency for session authentication.",
            "Verify cryptographic CPU utilization for signature verification.",
        ],
    }

    DEFAULT_ACTIONS = [
        "Review service stderr/stdout logs around the incident window.",
        "Check CPU, memory, and thread utilization against allocated limits.",
        "Verify downstream network connectivity and latency.",
        "Inspect recent code deployments and configuration rollouts.",
    ]

    def __init__(self):
        pass

    def build_candidate_bundle(
        self,
        candidate: CandidateRanking,
        df_metrics: Optional[pd.DataFrame],
        df_logs: Optional[pd.DataFrame],
        df_traces: Optional[pd.DataFrame],
        anomalies: List[AnomalyRecord],
        graph_result: GraphAnalysisResult,
    ) -> EvidenceBundle:
        """
        Builds a comprehensive, grounded evidence bundle for a candidate service.
        """
        svc = candidate.service
        observed_facts: List[Dict[str, Any]] = []
        algorithmic_inferences: List[Dict[str, Any]] = []
        cited_ids: List[str] = list(candidate.supporting_evidence_ids)

        # 1. Collect Metrics Evidence
        metrics_summary: List[Dict[str, Any]] = []
        if df_metrics is not None and not df_metrics.empty and "service" in df_metrics.columns:
            svc_metrics = df_metrics[df_metrics["service"] == svc]
            for idx, row in svc_metrics.iterrows():
                val = float(row.get("metric_value", 0.0))
                m_name = str(row.get("metric_name", ""))
                ts = str(row.get("timestamp", ""))
                ev_id = f"METRIC-{svc}-{m_name}-{idx}"

                metrics_summary.append({
                    "evidence_id": ev_id,
                    "metric_name": m_name,
                    "value": val,
                    "timestamp": ts,
                })

        # Match with detected metric anomalies
        for anom in anomalies:
            if anom.service == svc:
                if anom.evidence_reference not in cited_ids:
                    cited_ids.append(anom.evidence_reference)

                observed_facts.append({
                    "evidence_id": anom.evidence_reference,
                    "type": "METRIC_OBSERVATION",
                    "description": f"Observed {anom.metric} = {anom.observed_value:.2f} (baseline was {anom.baseline_value:.2f}).",
                    "timestamp": anom.timestamp,
                })

                z_info = f" (robust z-score = {anom.z_score:.2f})" if anom.z_score is not None else ""
                algorithmic_inferences.append({
                    "evidence_id": anom.evidence_reference,
                    "type": "ANOMALY_DETECTION",
                    "inference": f"{anom.severity} anomaly detected in {anom.metric} via {anom.detection_method}{z_info} with anomaly score {anom.anomaly_score:.2f}.",
                })

        # 2. Collect Logs Evidence
        logs_summary: List[Dict[str, Any]] = []
        if df_logs is not None and not df_logs.empty and "service" in df_logs.columns:
            svc_logs = df_logs[df_logs["service"] == svc]
            for idx, row in svc_logs.iterrows():
                log_id = str(row.get("log_id", f"LOG-{svc}-{idx}"))
                lvl = str(row.get("level", "INFO")).upper().strip()
                msg = str(row.get("message", ""))
                ts = str(row.get("timestamp", ""))

                log_entry = {
                    "evidence_id": log_id,
                    "level": lvl,
                    "message": msg,
                    "timestamp": ts,
                }
                logs_summary.append(log_entry)

                if lvl in ["ERROR", "FATAL", "CRITICAL"]:
                    if log_id not in cited_ids:
                        cited_ids.append(log_id)
                    observed_facts.append({
                        "evidence_id": log_id,
                        "type": "ERROR_LOG",
                        "description": f"[{lvl}] {msg}",
                        "timestamp": ts,
                    })

        # 3. Collect Traces Evidence
        traces_summary: List[Dict[str, Any]] = []
        if df_traces is not None and not df_traces.empty:
            dest_col = "destination_service" if "destination_service" in df_traces.columns else "service"
            src_col = "source_service" if "source_service" in df_traces.columns else None

            if dest_col in df_traces.columns:
                svc_traces = df_traces[df_traces[dest_col] == svc]
                for idx, row in svc_traces.iterrows():
                    tr_id = str(row.get("trace_id", f"TRACE-{idx}"))
                    lat = float(row.get("latency_ms", 0.0))
                    src = str(row.get(src_col, "client")) if src_col else "client"
                    ts = str(row.get("timestamp", ""))

                    traces_summary.append({
                        "evidence_id": tr_id,
                        "source": src,
                        "destination": svc,
                        "latency_ms": lat,
                        "timestamp": ts,
                    })

                    if lat >= 350.0:
                        if tr_id not in cited_ids:
                            cited_ids.append(tr_id)
                        observed_facts.append({
                            "evidence_id": tr_id,
                            "type": "TRACE_SPAN_LATENCY",
                            "description": f"Trace {tr_id}: Call from {src} to {svc} experienced high latency ({lat:.1f}ms).",
                            "timestamp": ts,
                        })

        # 4. Dependency Context
        node_info = graph_result.nodes.get(svc)
        dep_summary = {
            "service": svc,
            "callers": node_info.callers if node_info else [],
            "dependencies": node_info.dependencies if node_info else [],
            "is_isolated": node_info.is_isolated if node_info else True,
            "in_degree": node_info.in_degree if node_info else 0,
            "out_degree": node_info.out_degree if node_info else 0,
        }

        if node_info and node_info.callers:
            algorithmic_inferences.append({
                "evidence_id": f"DEP-{svc}-impact",
                "type": "DEPENDENCY_IMPACT",
                "inference": f"{svc} is called by {len(node_info.callers)} upstream service(s): {', '.join(node_info.callers)}. Failures here propagate upstream.",
            })

        # 5. Recommended Actions
        actions = self.RECOMMENDED_ACTIONS_MAP.get(svc, self.DEFAULT_ACTIONS)

        return EvidenceBundle(
            service=svc,
            rank=candidate.rank,
            score=candidate.score,
            ranking_method=candidate.ranking_method,
            observed_facts=observed_facts,
            algorithmic_inferences=algorithmic_inferences,
            recommended_actions=actions,
            cited_evidence_ids=cited_ids,
            metrics_summary=metrics_summary,
            logs_summary=logs_summary,
            traces_summary=traces_summary,
            dependencies_summary=dep_summary,
        )

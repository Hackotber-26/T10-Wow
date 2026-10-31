"""
RootSight - End-to-End Outage Diagnosis Pipeline
Coordinates: Data Ingestion -> Anomaly Detection -> Graph Construction ->
Deterministic Ranking -> Evidence Collection -> Evidence Verification -> Report Generation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
import pandas as pd

from detector.anomaly import AnomalyDetector, AnomalyRecord
from engine.graph_builder import DependencyGraphBuilder, GraphAnalysisResult
from engine.ranking import DeterministicRankingEngine, CandidateRanking
from reporting.evidence import EvidenceCollector, EvidenceBundle
from reporting.verifier import EvidenceVerifier, VerificationResult
from reporting.narrator import DiagnosisNarrator, DiagnosisReport


@dataclass
class PipelineResult:
    incident_id: str
    services: List[str]
    anomalies: List[AnomalyRecord]
    graph_result: GraphAnalysisResult
    candidates: List[CandidateRanking]
    top_candidate: Optional[CandidateRanking]
    bundles: Dict[str, EvidenceBundle]
    verification_result: VerificationResult
    report: DiagnosisReport
    method_used: str
    execution_summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "services": self.services,
            "anomaly_count": len(self.anomalies),
            "total_candidates": len(self.candidates),
            "top_candidate": self.top_candidate.to_dict() if self.top_candidate else None,
            "verification_status": self.verification_result.audit_status,
            "method_used": self.method_used,
            "report": self.report.to_dict(),
        }


class DiagnosisPipeline:
    """
    Executes the end-to-end RootSight root-cause diagnosis pipeline.
    """

    def __init__(
        self,
        mad_threshold: float = 3.0,
        min_history_observations: int = 4,
        weight_metric: float = 0.35,
        weight_dependency: float = 0.30,
        weight_trace: float = 0.20,
        weight_log: float = 0.15,
        damping_factor: float = 0.85,
    ):
        self.detector = AnomalyDetector(
            mad_threshold=mad_threshold,
            min_history_observations=min_history_observations,
        )
        self.graph_builder = DependencyGraphBuilder()
        self.ranking_engine = DeterministicRankingEngine(
            weight_metric=weight_metric,
            weight_dependency=weight_dependency,
            weight_trace=weight_trace,
            weight_log=weight_log,
            damping_factor=damping_factor,
        )
        self.evidence_collector = EvidenceCollector()
        self.verifier = EvidenceVerifier()
        self.narrator = DiagnosisNarrator()

    def run(
        self,
        incident_id: str,
        df_metrics: Optional[pd.DataFrame] = None,
        df_logs: Optional[pd.DataFrame] = None,
        df_traces: Optional[pd.DataFrame] = None,
        df_dependencies: Optional[pd.DataFrame] = None,
        ranking_method: str = "baseline",  # 'baseline' or 'pagerank'
        prefer_ai_narration: bool = False,
    ) -> PipelineResult:
        """
        Runs the full deterministic RCA pipeline.
        """
        # 1. Collect all known services across inputs
        known_services: Set[str] = set()
        if df_metrics is not None and not df_metrics.empty and "service" in df_metrics.columns:
            known_services.update(df_metrics["service"].dropna().astype(str).str.strip().unique())
        if df_logs is not None and not df_logs.empty and "service" in df_logs.columns:
            known_services.update(df_logs["service"].dropna().astype(str).str.strip().unique())
        if df_traces is not None and not df_traces.empty:
            if "source_service" in df_traces.columns:
                known_services.update(df_traces["source_service"].dropna().astype(str).str.strip().unique())
            if "destination_service" in df_traces.columns:
                known_services.update(df_traces["destination_service"].dropna().astype(str).str.strip().unique())
        if df_dependencies is not None and not df_dependencies.empty:
            src_col = "source_service" if "source_service" in df_dependencies.columns else "source"
            dst_col = "destination_service" if "destination_service" in df_dependencies.columns else "destination"
            if src_col in df_dependencies.columns:
                known_services.update(df_dependencies[src_col].dropna().astype(str).str.strip().unique())
            if dst_col in df_dependencies.columns:
                known_services.update(df_dependencies[dst_col].dropna().astype(str).str.strip().unique())

        known_services.discard("")

        # 2. Anomaly Detection
        anomalies: List[AnomalyRecord] = []
        metric_anoms = self.detector.detect_metric_anomalies(df_metrics)
        log_anoms = self.detector.detect_log_anomalies(df_logs)
        trace_anoms = self.detector.detect_trace_anomalies(df_traces)

        anomalies.extend(metric_anoms)
        anomalies.extend(log_anoms)
        anomalies.extend(trace_anoms)

        # Build anomaly summary map for graph nodes
        anom_map: Dict[str, Dict[str, Any]] = {}
        for a in anomalies:
            if a.service not in anom_map:
                anom_map[a.service] = {"is_anomalous": True, "count": 0, "max_severity": "INFO"}
            entry = anom_map[a.service]
            entry["count"] += 1
            if a.severity == "CRITICAL":
                entry["max_severity"] = "CRITICAL"
            elif a.severity == "WARNING" and entry["max_severity"] != "CRITICAL":
                entry["max_severity"] = "WARNING"

        # 3. Construct Service Dependency Graph
        graph_result = self.graph_builder.build_graph(
            df_traces=df_traces,
            df_dependencies=df_dependencies,
            known_services=known_services,
            anomalous_services=anom_map,
        )

        # 4. Deterministic Root-Cause Ranking
        all_services = set(graph_result.nodes.keys()).union(known_services)
        if ranking_method.lower() in ["pagerank", "personalized_pagerank", "method_b"]:
            candidates = self.ranking_engine.rank_personalized_pagerank(
                all_services, anomalies, graph_result
            )
            method_name = "Personalized PageRank"
        else:
            candidates = self.ranking_engine.rank_explainable_baseline(
                all_services, anomalies, graph_result
            )
            method_name = "Explainable Baseline"

        top_candidate = candidates[0] if candidates else None

        # 5. Evidence Collection
        bundles: Dict[str, EvidenceBundle] = {}
        for cand in candidates[:5]:  # Top 5 candidates get complete evidence bundles
            bundle = self.evidence_collector.build_candidate_bundle(
                candidate=cand,
                df_metrics=df_metrics,
                df_logs=df_logs,
                df_traces=df_traces,
                anomalies=anomalies,
                graph_result=graph_result,
            )
            bundles[cand.service] = bundle

        # 6. Evidence Verification
        if top_candidate and top_candidate.service in bundles:
            top_bundle = bundles[top_candidate.service]
            verification_result = self.verifier.verify_bundle(
                bundle=top_bundle,
                df_metrics=df_metrics,
                df_logs=df_logs,
                df_traces=df_traces,
            )
        else:
            verification_result = VerificationResult(
                passed=False,
                total_checked=0,
                passed_count=0,
                failed_count=0,
                checks=["No candidates available to verify."],
                failure_log=[],
                audit_status="NO_DATA",
            )

        # 7. Diagnosis Report Generation
        report = self.narrator.generate_report(
            incident_id=incident_id,
            candidates=candidates,
            bundles=bundles,
            verification_result=verification_result,
            prefer_ai=prefer_ai_narration,
        )

        # Summary metadata
        summary_meta = {
            "incident_id": incident_id,
            "total_services": len(all_services),
            "total_anomalies": len(anomalies),
            "metric_anomalies": len(metric_anoms),
            "log_anomalies": len(log_anoms),
            "trace_anomalies": len(trace_anoms),
            "ranking_method": method_name,
            "top_candidate": top_candidate.service if top_candidate else "None",
            "top_score": top_candidate.score if top_candidate else 0.0,
            "verification_status": verification_result.audit_status,
        }

        return PipelineResult(
            incident_id=incident_id,
            services=sorted(list(all_services)),
            anomalies=anomalies,
            graph_result=graph_result,
            candidates=candidates,
            top_candidate=top_candidate,
            bundles=bundles,
            verification_result=verification_result,
            report=report,
            method_used=method_name,
            execution_summary=summary_meta,
        )

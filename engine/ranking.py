"""
RootSight - Deterministic Root-Cause Ranking Engine
Implements:
  Method A: Explainable Weighted Scoring Baseline S(v) = wA*A(v) + wD*D(v) + wT*T(v) + wL*L(v)
  Method B: Personalized PageRank over Service Dependency Topology
Produces deterministic, auditable rankings with stable tie-breaking and uncertainty estimation.
"""

from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Set
import networkx as nx
import numpy as np

from detector.anomaly import AnomalyRecord
from engine.graph_builder import GraphAnalysisResult


@dataclass
class CandidateRanking:
    service: str
    rank: int
    score: float
    ranking_method: str
    metric_signal: float
    dependency_signal: float
    trace_signal: float
    log_signal: float
    supporting_evidence_ids: List[str]
    concise_reason: str
    evidence_quality: str  # 'HIGH_CONFIDENCE', 'MODERATE_CONFIDENCE', 'LOW_CONFIDENCE', 'INSUFFICIENT_EVIDENCE'
    component_breakdown: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DeterministicRankingEngine:
    """
    Ranks potential root causes deterministically without LLM intervention.
    """

    def __init__(
        self,
        weight_metric: float = 0.35,
        weight_dependency: float = 0.30,
        weight_trace: float = 0.20,
        weight_log: float = 0.15,
        damping_factor: float = 0.85,
    ):
        self.w_metric = float(weight_metric)
        self.w_dep = float(weight_dependency)
        self.w_trace = float(weight_trace)
        self.w_log = float(weight_log)
        self.damping_factor = float(damping_factor)

        # Normalize weights so sum is 1.0
        total_w = self.w_metric + self.w_dep + self.w_trace + self.w_log
        if total_w > 0:
            self.w_metric /= total_w
            self.w_dep /= total_w
            self.w_trace /= total_w
            self.w_log /= total_w

    def _extract_signals(
        self,
        services: Set[str],
        anomalies: List[AnomalyRecord],
        graph_result: GraphAnalysisResult,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extracts normalized features A(v), D(v), T(v), L(v) and evidence IDs for each service.
        """
        signals: Dict[str, Dict[str, Any]] = {
            s: {
                "A": 0.0,  # Metric anomaly
                "D": 0.0,  # Dependency causal impact
                "T": 0.0,  # Trace anomaly
                "L": 0.0,  # Log anomaly
                "metric_records": [],
                "trace_records": [],
                "log_records": [],
                "evidence_ids": [],
            }
            for s in services
        }

        # 1. Bucket anomalies by service and category
        for anom in anomalies:
            svc = anom.service
            if svc not in signals:
                signals[svc] = {
                    "A": 0.0, "D": 0.0, "T": 0.0, "L": 0.0,
                    "metric_records": [], "trace_records": [], "log_records": [],
                    "evidence_ids": [],
                }
            
            sig = signals[svc]
            sig["evidence_ids"].append(anom.evidence_reference)

            if anom.detection_method == "log_error_frequency" or "log" in anom.metric:
                sig["L"] = max(sig["L"], anom.anomaly_score)
                sig["log_records"].append(anom)
            elif "trace" in anom.metric:
                sig["T"] = max(sig["T"], anom.anomaly_score)
                sig["trace_records"].append(anom)
            else:
                # Metric anomalies: take max or average of top anomalies
                sig["A"] = max(sig["A"], anom.anomaly_score)
                sig["metric_records"].append(anom)

        # 2. Compute Dependency Causal Evidence D(v)
        # In service dependencies A -> B (A calls B):
        # If callee B has an anomaly, and upstream callers of B also experience anomalies,
        # B is explaining those upstream anomalies. Thus B receives a high causal score.
        G = graph_result.graph
        anomalous_nodes = {s for s, sig in signals.items() if (sig["A"] > 0 or sig["T"] > 0 or sig["L"] > 0)}

        for svc in signals:
            if svc in G:
                # Upstream callers of svc that are also anomalous
                try:
                    ancestors = set(nx.ancestors(G, svc))
                except Exception:
                    ancestors = set()
                impacted_anom_callers = ancestors.intersection(anomalous_nodes)
                
                # Callee failure explanation power:
                # If svc is anomalous, and 2 upstream callers are also anomalous, D(svc) increases.
                # If svc calls downstream callees that are anomalous, then svc might be a victim, not root!
                try:
                    descendants = set(nx.descendants(G, svc))
                except Exception:
                    descendants = set()
                failing_dependencies = descendants.intersection(anomalous_nodes)

                svc_is_anom = (signals[svc]["A"] > 0 or signals[svc]["T"] > 0 or signals[svc]["L"] > 0)
                if svc_is_anom:
                    # Positive score for explaining upstream caller failures
                    upstream_boost = len(impacted_anom_callers) * 0.35
                    # Penalty if downstream dependencies are also failing (victim of downstream failure)
                    downstream_penalty = len(failing_dependencies) * 0.25
                    dep_score = max(0.0, min(1.0, 0.2 + upstream_boost - downstream_penalty))
                    signals[svc]["D"] = dep_score
                else:
                    signals[svc]["D"] = 0.0

        return signals

    def rank_explainable_baseline(
        self,
        services: Set[str],
        anomalies: List[AnomalyRecord],
        graph_result: GraphAnalysisResult,
    ) -> List[CandidateRanking]:
        """
        Method A: Transparent weighted scoring baseline
        S(v) = wA * A(v) + wD * D(v) + wT * T(v) + wL * L(v)
        Deterministic tie-breaking: (-score, -A, -L, -D, service_name)
        """
        signals = self._extract_signals(services, anomalies, graph_result)
        candidates: List[CandidateRanking] = []

        total_anomalies = len(anomalies)

        for svc, sig in signals.items():
            A = sig["A"]
            D = sig["D"]
            T = sig["T"]
            L = sig["L"]

            # Weighted linear combination
            score = (self.w_metric * A) + (self.w_dep * D) + (self.w_trace * T) + (self.w_log * L)
            score = round(float(np.clip(score, 0.0, 1.0)), 4)

            # Evidence Quality Assessment
            ev_count = len(sig["evidence_ids"])
            if total_anomalies == 0 or (score < 0.05 and ev_count == 0):
                quality = "INSUFFICIENT_EVIDENCE" if total_anomalies == 0 else "LOW_CONFIDENCE"
            elif ev_count >= 3 and score >= 0.50:
                quality = "HIGH_CONFIDENCE"
            elif ev_count >= 1 and score >= 0.25:
                quality = "MODERATE_CONFIDENCE"
            else:
                quality = "LOW_CONFIDENCE"

            # Reason formulation based strictly on observed components
            reasons = []
            if A > 0.4:
                reasons.append(f"severe metric anomalies (A={A:.2f})")
            if D > 0.3:
                reasons.append(f"high downstream fault propagation impact (D={D:.2f})")
            if T > 0.4:
                reasons.append(f"significant trace latency degradation (T={T:.2f})")
            if L > 0.3:
                reasons.append(f"critical log error rate surge (L={L:.2f})")

            concise_reason = (
                f"Candidate identified by {', '.join(reasons)}"
                if reasons else "Minimal or no anomalous telemetry observed"
            )

            candidates.append(
                CandidateRanking(
                    service=svc,
                    rank=0,  # Populated after sort
                    score=score,
                    ranking_method="Explainable Baseline (Weighted Signals)",
                    metric_signal=round(A, 4),
                    dependency_signal=round(D, 4),
                    trace_signal=round(T, 4),
                    log_signal=round(L, 4),
                    supporting_evidence_ids=sig["evidence_ids"],
                    concise_reason=concise_reason,
                    evidence_quality=quality,
                    component_breakdown={"wA*A": round(self.w_metric * A, 4), "wD*D": round(self.w_dep * D, 4), "wT*T": round(self.w_trace * T, 4), "wL*L": round(self.w_log * L, 4)},
                )
            )

        # Stable deterministic sort:
        # Sort key: (-score, -metric_signal, -log_signal, -dependency_signal, service_name)
        candidates.sort(
            key=lambda c: (-c.score, -c.metric_signal, -c.log_signal, -c.dependency_signal, c.service)
        )

        # Assign 1-indexed ranks
        for idx, cand in enumerate(candidates, start=1):
            cand.rank = idx

        return candidates

    def rank_personalized_pagerank(
        self,
        services: Set[str],
        anomalies: List[AnomalyRecord],
        graph_result: GraphAnalysisResult,
    ) -> List[CandidateRanking]:
        """
        Method B: Personalized PageRank over Service Dependency Graph.
        
        Graph Orientation Semantics:
        Edges A -> B indicate service A calls service B.
        When B develops an internal failure (root cause), callers like A experience symptoms
        (latency spikes, timeouts).
        We construct a personalization vector on nodes with observed anomalies.
        Random walks traverse dependency edges towards invoked services (from caller to callee),
        so probability mass accumulates at the structural sink / callee bottleneck causing the failures.
        """
        signals = self._extract_signals(services, anomalies, graph_result)
        G = graph_result.graph.copy()

        # Ensure all services exist in graph
        for s in services:
            if not G.has_node(s):
                G.add_node(s)

        if len(G) == 0:
            # Fallback to baseline
            return self.rank_explainable_baseline(services, anomalies, graph_result)

        # Construct personalization vector from combined anomaly evidence
        personalization: Dict[str, float] = {}
        for s in G.nodes():
            sig = signals.get(s, {"A": 0.0, "T": 0.0, "L": 0.0})
            raw_evidence = (0.45 * sig.get("A", 0.0)) + (0.35 * sig.get("T", 0.0)) + (0.20 * sig.get("L", 0.0))
            # Base small epsilon to avoid zero division and guarantee full graph reachability
            personalization[s] = max(0.001, raw_evidence)

        total_pers = sum(personalization.values())
        if total_pers > 0:
            personalization = {k: v / total_pers for k, v in personalization.items()}

        # Adjust transition edge weights: calls to anomalous services carry higher transition probability
        for u, v in G.edges():
            v_sig = signals.get(v, {})
            v_anom = max(v_sig.get("A", 0.0), v_sig.get("T", 0.0), v_sig.get("L", 0.0))
            G[u][v]["weight"] = float(G[u][v].get("weight", 1.0)) * (1.0 + 4.0 * v_anom)

        # Add self-loops to sink nodes so causal random walk concentrates at terminal bottlenecks
        for n in list(G.nodes()):
            if G.out_degree(n) == 0:
                G.add_edge(n, n, weight=2.0)

        # Compute PageRank
        try:
            if G.number_of_edges() > 0:
                pr_scores = nx.pagerank(
                    G,
                    alpha=self.damping_factor,
                    personalization=personalization,
                    weight="weight",
                    max_iter=200,
                    tol=1e-6,
                )
            else:
                pr_scores = personalization
        except Exception:
            pr_scores = personalization

        # Scale raw PageRank scores to [0.0, 1.0]
        max_pr = max(pr_scores.values()) if pr_scores else 1.0
        norm_pr_scores = {k: (v / max_pr) if max_pr > 0 else 0.0 for k, v in pr_scores.items()}

        candidates: List[CandidateRanking] = []
        total_anomalies = len(anomalies)

        for svc in services:
            pr_val = pr_scores.get(svc, 0.0)
            norm_pr = norm_pr_scores.get(svc, 0.0)
            sig = signals.get(svc, {"A": 0.0, "D": 0.0, "T": 0.0, "L": 0.0, "evidence_ids": []})
            
            # Intrinsic anomaly score combines direct metric, trace, and log evidence
            intrinsic_score = (0.40 * sig["A"]) + (0.35 * sig["T"]) + (0.25 * sig["L"])

            # Combined PageRank score: blends topological centrality with intrinsic anomaly evidence
            if total_anomalies > 0:
                score = round(float(np.clip((0.55 * norm_pr) + (0.45 * intrinsic_score), 0.0, 1.0)), 4)
            else:
                score = 0.0

            ev_count = len(sig["evidence_ids"])
            if total_anomalies == 0 or (score < 0.05 and ev_count == 0):
                quality = "INSUFFICIENT_EVIDENCE" if total_anomalies == 0 else "LOW_CONFIDENCE"
            elif ev_count >= 2 and score >= 0.60:
                quality = "HIGH_CONFIDENCE"
            elif score >= 0.30:
                quality = "MODERATE_CONFIDENCE"
            else:
                quality = "LOW_CONFIDENCE"

            pr_val = pr_scores.get(svc, 0.0)
            concise_reason = (
                f"Personalized PageRank node mass {pr_val:.4f} supported by {ev_count} telemetry evidence items"
                if ev_count > 0 else f"Structural PageRank centrality {pr_val:.4f} without direct anomalous evidence"
            )

            candidates.append(
                CandidateRanking(
                    service=svc,
                    rank=0,
                    score=score,
                    ranking_method="Personalized PageRank (Graph Centrality)",
                    metric_signal=round(sig["A"], 4),
                    dependency_signal=round(sig["D"], 4),
                    trace_signal=round(sig["T"], 4),
                    log_signal=round(sig["L"], 4),
                    supporting_evidence_ids=sig["evidence_ids"],
                    concise_reason=concise_reason,
                    evidence_quality=quality,
                    component_breakdown={"raw_pagerank": round(pr_val, 6), "normalized_score": score},
                )
            )

        # Stable deterministic sort
        candidates.sort(
            key=lambda c: (-c.score, -c.metric_signal, -c.log_signal, c.service)
        )

        for idx, cand in enumerate(candidates, start=1):
            cand.rank = idx

        return candidates

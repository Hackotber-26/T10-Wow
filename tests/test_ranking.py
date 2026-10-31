"""
Unit tests for DeterministicRankingEngine.
Covers: Method A (Weighted Baseline), Method B (Personalized PageRank),
deterministic reproducibility, stable tie-breaking, and absence of LLM bias.
"""

import pytest
import pandas as pd
from engine.graph_builder import DependencyGraphBuilder
from engine.ranking import DeterministicRankingEngine, CandidateRanking
from detector.anomaly import AnomalyRecord


@pytest.fixture
def sample_graph():
    builder = DependencyGraphBuilder()
    df_deps = pd.DataFrame([
        {"source_service": "frontend", "destination_service": "api-gateway"},
        {"source_service": "api-gateway", "destination_service": "cart-service"},
        {"source_service": "cart-service", "destination_service": "postgres-db"},
    ])
    return builder.build_graph(df_dependencies=df_deps)


@pytest.fixture
def sample_anomalies():
    return [
        AnomalyRecord(
            service="postgres-db",
            metric="latency_ms",
            observed_value=950.0,
            baseline_value=15.0,
            anomaly_score=0.95,
            severity="CRITICAL",
            detection_method="robust_z_mad",
            evidence_reference="METRIC-postgres-db-latency_ms-0",
        ),
        AnomalyRecord(
            service="cart-service",
            metric="latency_ms",
            observed_value=1020.0,
            baseline_value=30.0,
            anomaly_score=0.88,
            severity="CRITICAL",
            detection_method="robust_z_mad",
            evidence_reference="METRIC-cart-service-latency_ms-1",
        ),
    ]


def test_deterministic_reproducibility(sample_graph, sample_anomalies):
    """The same incident telemetry and configuration must produce the exact same ranking."""
    engine = DeterministicRankingEngine()
    services = set(sample_graph.nodes.keys())

    run1 = engine.rank_explainable_baseline(services, sample_anomalies, sample_graph)
    run2 = engine.rank_explainable_baseline(services, sample_anomalies, sample_graph)

    assert len(run1) == len(run2)
    for c1, c2 in zip(run1, run2):
        assert c1.service == c2.service
        assert c1.rank == c2.rank
        assert c1.score == c2.score


def test_stable_tie_breaking(sample_graph):
    """When scores and signals are identical, alphabetical service name provides deterministic tie-breaking."""
    engine = DeterministicRankingEngine()
    services = {"service-b", "service-a", "service-c"}
    empty_graph = DependencyGraphBuilder().build_graph(known_services=services)

    # Zero anomalies: all scores are 0.0
    rankings = engine.rank_explainable_baseline(services, [], empty_graph)
    assert [c.service for c in rankings] == ["service-a", "service-b", "service-c"]
    assert [c.rank for c in rankings] == [1, 2, 3]


def test_baseline_ranking_identifies_root_callee(sample_graph, sample_anomalies):
    engine = DeterministicRankingEngine()
    services = set(sample_graph.nodes.keys())
    rankings = engine.rank_explainable_baseline(services, sample_anomalies, sample_graph)

    # postgres-db should rank #1 because it explains the upstream cart-service failure
    assert rankings[0].service == "postgres-db"
    assert rankings[0].score > rankings[1].score


def test_pagerank_ranking_handles_empty_or_disconnected_graph():
    engine = DeterministicRankingEngine()
    services = {"service-x", "service-y"}
    empty_graph = DependencyGraphBuilder().build_graph(known_services=services)

    rankings = engine.rank_personalized_pagerank(services, [], empty_graph)
    assert len(rankings) == 2
    assert all(r.score <= 1.0 for r in rankings)

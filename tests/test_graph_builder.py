"""
Unit tests for DependencyGraphBuilder.
Covers: trace graph construction, explicit topologies, isolated nodes,
disconnected graphs, empty graphs, and upstream/downstream impact paths.
"""

import pytest
import pandas as pd
import networkx as nx
from engine.graph_builder import DependencyGraphBuilder


def test_empty_graph_construction():
    builder = DependencyGraphBuilder()
    res = builder.build_graph(pd.DataFrame(), pd.DataFrame())
    assert res.total_services == 0
    assert res.total_dependencies == 0
    assert len(res.graph) == 0


def test_isolated_services_preserved():
    builder = DependencyGraphBuilder()
    known = {"svc-a", "svc-b", "isolated-worker"}
    df_deps = pd.DataFrame([
        {"source_service": "svc-a", "destination_service": "svc-b", "call_count": 100}
    ])
    res = builder.build_graph(df_dependencies=df_deps, known_services=known)
    assert res.total_services == 3
    assert "isolated-worker" in res.isolated_services
    assert res.nodes["isolated-worker"].is_isolated is True


def test_disconnected_dependency_graph():
    builder = DependencyGraphBuilder()
    df_deps = pd.DataFrame([
        {"source_service": "cluster1-a", "destination_service": "cluster1-b"},
        {"source_service": "cluster2-x", "destination_service": "cluster2-y"},
    ])
    res = builder.build_graph(df_dependencies=df_deps)
    assert res.total_services == 4
    assert res.total_dependencies == 2
    # Two separate components
    assert nx.number_weakly_connected_components(res.graph) == 2


def test_trace_graph_edge_aggregation():
    builder = DependencyGraphBuilder()
    df_traces = pd.DataFrame([
        {"source_service": "gateway", "destination_service": "cart", "latency_ms": 50.0},
        {"source_service": "gateway", "destination_service": "cart", "latency_ms": 70.0},
        {"source_service": "cart", "destination_service": "db", "latency_ms": 10.0},
    ])
    res = builder.build_graph(df_traces=df_traces)
    assert res.total_services == 3
    assert res.total_dependencies == 2
    assert res.graph["gateway"]["cart"]["call_count"] == 2
    assert res.graph["gateway"]["cart"]["total_latency"] == 120.0


def test_upstream_and_downstream_impact():
    builder = DependencyGraphBuilder()
    df_deps = pd.DataFrame([
        {"source_service": "frontend", "destination_service": "gateway"},
        {"source_service": "gateway", "destination_service": "cart"},
        {"source_service": "cart", "destination_service": "postgres"},
    ])
    res = builder.build_graph(df_dependencies=df_deps)
    upstream_of_postgres = builder.get_upstream_impact(res.graph, "postgres")
    assert set(upstream_of_postgres) == {"frontend", "gateway", "cart"}

    downstream_of_gateway = builder.get_downstream_dependencies(res.graph, "gateway")
    assert set(downstream_of_gateway) == {"cart", "postgres"}

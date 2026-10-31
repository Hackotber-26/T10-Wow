"""
RootSight - Service Dependency Graph Builder
Constructs directed dependency graphs using NetworkX from distributed traces and explicit topologies.
Calculates topological properties, upstream/downstream impact paths, and serializes for visualization.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
import networkx as nx
import pandas as pd


@dataclass
class ServiceNodeInfo:
    service: str
    in_degree: int
    out_degree: int
    callers: List[str]  # Upstream callers (services that call this service)
    dependencies: List[str]  # Downstream dependencies (services this service calls)
    is_isolated: bool
    is_anomalous: bool = False
    anomaly_count: int = 0
    max_severity: str = "HEALTHY"


@dataclass
class GraphAnalysisResult:
    graph: nx.DiGraph
    nodes: Dict[str, ServiceNodeInfo]
    edge_list: List[Dict[str, Any]]
    isolated_services: List[str]
    root_cause_candidates_subgraph: List[str]
    total_services: int
    total_dependencies: int

    def to_serializable(self) -> Dict[str, Any]:
        return {
            "total_services": self.total_services,
            "total_dependencies": self.total_dependencies,
            "isolated_services": self.isolated_services,
            "nodes": {k: v.__dict__ for k, v in self.nodes.items()},
            "edges": self.edge_list,
        }


class DependencyGraphBuilder:
    """
    Builds and analyzes directed service dependency graphs.
    Edge convention:
      A -> B indicates service A invokes / depends on service B.
      A is the upstream caller; B is the downstream callee.
      When B fails, latency or errors propagate back to A.
    """

    def __init__(self):
        pass

    def build_graph(
        self,
        df_traces: Optional[pd.DataFrame] = None,
        df_dependencies: Optional[pd.DataFrame] = None,
        known_services: Optional[Set[str]] = None,
        anomalous_services: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> GraphAnalysisResult:
        """
        Constructs a directed graph nx.DiGraph from traces and/or explicit dependencies.
        Ensures all known services (even disconnected or isolated) are represented as nodes.
        """
        G = nx.DiGraph()

        # 1. Register all known services first (guarantees isolated nodes appear)
        all_services: Set[str] = set()
        if known_services:
            all_services.update(str(s).strip() for s in known_services if str(s).strip())

        # 2. Add edges from explicit dependency list if available
        if df_dependencies is not None and not df_dependencies.empty:
            for _, row in df_dependencies.iterrows():
                src = str(row.get("source_service", row.get("source", ""))).strip()
                dst = str(row.get("destination_service", row.get("destination", ""))).strip()
                if src and dst and src != dst:
                    all_services.add(src)
                    all_services.add(dst)
                    weight = float(row.get("call_count", row.get("weight", 1.0)))
                    call_cnt = int(row.get("call_count", 1))
                    if G.has_edge(src, dst):
                        G[src][dst]["weight"] += weight
                        G[src][dst]["call_count"] = G[src][dst].get("call_count", 0) + call_cnt
                    else:
                        G.add_edge(
                            src,
                            dst,
                            weight=weight,
                            call_count=call_cnt,
                            total_latency=0.0,
                            source_type="explicit",
                        )

        # 3. Add edges and aggregate weights from traces
        if df_traces is not None and not df_traces.empty:
            cols = df_traces.columns
            src_col = "source_service" if "source_service" in cols else "source"
            dst_col = "destination_service" if "destination_service" in cols else "destination"
            lat_col = "latency_ms" if "latency_ms" in cols else None

            if src_col in cols and dst_col in cols:
                for _, row in df_traces.iterrows():
                    src = str(row[src_col]).strip()
                    dst = str(row[dst_col]).strip()
                    if src and dst and src != dst:
                        all_services.add(src)
                        all_services.add(dst)
                        lat = float(row[lat_col]) if lat_col and pd.notnull(row[lat_col]) else 0.0

                        if G.has_edge(src, dst):
                            G[src][dst]["call_count"] = G[src][dst].get("call_count", 0) + 1
                            G[src][dst]["total_latency"] = G[src][dst].get("total_latency", 0.0) + lat
                            G[src][dst]["weight"] = G[src][dst]["call_count"]
                        else:
                            G.add_edge(
                                src,
                                dst,
                                call_count=1,
                                total_latency=lat,
                                weight=1.0,
                                source_type="trace",
                            )

        # Ensure all services exist in the graph as nodes
        for svc in all_services:
            if not G.has_node(svc):
                G.add_node(svc)

        # 4. Compute node metadata
        anom_map = anomalous_services or {}
        nodes_info: Dict[str, ServiceNodeInfo] = {}
        isolated: List[str] = []

        for node in G.nodes():
            in_deg = G.in_degree(node)
            out_deg = G.out_degree(node)
            # Callers: predecessors in A -> B (A calls B, so A is predecessor of B)
            callers = list(G.predecessors(node))
            # Dependencies: successors in A -> B (B is successor of A)
            dependencies = list(G.successors(node))
            is_iso = (in_deg == 0 and out_deg == 0)
            if is_iso:
                isolated.append(node)

            svc_anom = anom_map.get(node, {})
            is_anom = bool(svc_anom.get("is_anomalous", False))
            anom_cnt = int(svc_anom.get("count", 0))
            max_sev = str(svc_anom.get("max_severity", "HEALTHY"))

            nodes_info[node] = ServiceNodeInfo(
                service=node,
                in_degree=in_deg,
                out_degree=out_deg,
                callers=callers,
                dependencies=dependencies,
                is_isolated=is_iso,
                is_anomalous=is_anom,
                anomaly_count=anom_cnt,
                max_severity=max_sev,
            )

        # 5. Format edges for frontend serialization
        edge_list: List[Dict[str, Any]] = []
        for u, v, data in G.edges(data=True):
            avg_lat = 0.0
            call_cnt = data.get("call_count", 1)
            if "total_latency" in data and call_cnt > 0:
                avg_lat = round(data["total_latency"] / call_cnt, 2)
            edge_list.append({
                "source": u,
                "target": v,
                "weight": data.get("weight", 1.0),
                "call_count": call_cnt,
                "avg_latency_ms": avg_lat,
                "source_type": data.get("source_type", "inferred"),
            })

        # Identify subgraphs between anomalous services
        anom_nodes = [n for n, info in nodes_info.items() if info.is_anomalous]

        return GraphAnalysisResult(
            graph=G,
            nodes=nodes_info,
            edge_list=edge_list,
            isolated_services=isolated,
            root_cause_candidates_subgraph=anom_nodes,
            total_services=len(all_services),
            total_dependencies=len(edge_list),
        )

    @staticmethod
    def get_upstream_impact(G: nx.DiGraph, service: str) -> List[str]:
        """
        When 'service' experiences an outage or latency spike, its upstream callers
        (all nodes that can reach 'service') are impacted.
        Returns all upstream services that depend directly or transitively on 'service'.
        """
        if service not in G:
            return []
        # In G (A -> B means A calls B), upstream callers can reach 'service'.
        # We find ancestors of 'service' in G:
        try:
            return list(nx.ancestors(G, service))
        except Exception:
            return []

    @staticmethod
    def get_downstream_dependencies(G: nx.DiGraph, service: str) -> List[str]:
        """
        Returns all downstream services called directly or transitively by 'service'.
        """
        if service not in G:
            return []
        try:
            return list(nx.descendants(G, service))
        except Exception:
            return []

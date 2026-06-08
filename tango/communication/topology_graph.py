"""Topology-Adaptive Communication Graph builder."""

import networkx as nx
from typing import Dict, List, Tuple, Set
from collections import defaultdict


class TopologyAdaptiveGraph:
    """
    Dynamically constructs agent communication topology based on:
    1. Cross-community propagation strength (edge weight sum between communities)
    2. Boundary risk (proportion of boundary nodes)
    3. Redundancy (Jaccard overlap of neighbor sets)
    
    Only communities with sufficient connection weight establish communication links.
    """
    
    def __init__(self, 
                 alpha_propagation: float = 0.5,
                 beta_boundary: float = 0.3, 
                 gamma_redundancy: float = 0.2,
                 connection_threshold: float = 0.1):
        self.alpha = alpha_propagation
        self.beta = beta_boundary
        self.gamma = gamma_redundancy
        self.threshold = connection_threshold
        
    def build(self, 
              G: nx.Graph,
              communities: Dict[int, any],  # community_id -> Community object
              ) -> nx.Graph:
        """
        Build communication graph based on current network state.
        
        Returns:
            nx.Graph: Communication topology (nodes = community IDs, 
                      edges = communication links with weights)
        """
        comm_graph = nx.Graph()
        com_ids = list(communities.keys())
        comm_graph.add_nodes_from(com_ids)
        
        # Precompute node-to-community mapping
        node_to_com = {}
        for cid, com in communities.items():
            for node in com.state.nodes:
                node_to_com[node] = cid
        
        # Compute pairwise metrics
        for i, cid_a in enumerate(com_ids):
            for cid_b in com_ids[i+1:]:
                com_a = communities[cid_a]
                com_b = communities[cid_b]
                
                # 1. Propagation strength: sum of cross-community edge weights
                prop_strength = self._compute_propagation_strength(
                    G, com_a.state.nodes, com_b.state.nodes, node_to_com)
                
                # 2. Boundary risk: combined boundary exposure
                bnd_risk_a = len(com_a.state.boundary_nodes) / max(1, len(com_a.state.nodes))
                bnd_risk_b = len(com_b.state.boundary_nodes) / max(1, len(com_b.state.nodes))
                combined_boundary_risk = (bnd_risk_a + bnd_risk_b) / 2
                
                # 3. Redundancy: Jaccard of neighboring communities
                redundancy = self._compute_redundancy(
                    com_a.state.neighbor_community_ids, 
                    com_b.state.neighbor_community_ids)
                
                # Weighted score
                score = (self.alpha * prop_strength + 
                        self.beta * combined_boundary_risk + 
                        self.gamma * redundancy)
                
                if score >= self.threshold:
                    comm_graph.add_edge(cid_a, cid_b, weight=score,
                                       propagation_strength=prop_strength,
                                       boundary_risk=combined_boundary_risk)
        
        return comm_graph
    
    def _compute_propagation_strength(self, G, nodes_a, nodes_b, node_to_com) -> float:
        """Sum of edge weights crossing between two communities."""
        total_weight = 0.0
        
        for u in nodes_a:
            for v in G.neighbors(u):
                if v in node_to_com and node_to_com[v] not in [node_to_com.get(u)]:
                    # v belongs to a different community
                    if v in nodes_b or node_to_com.get(v) == node_to_com.get(
                            next(n for n in nodes_b if n in node_to_com)):
                        total_weight += G[u][v].get('weight', 1.0)
        
        # Normalize by total possible
        max_possible = len(nodes_a) + len(nodes_b)
        if max_possible == 0:
            return 0.0
        return min(1.0, total_weight / max_possible)
    
    def _compute_redundancy(self, neighbors_a: List[int], neighbors_b: List[int]) -> float:
        """Jaccard similarity of neighbor community sets."""
        set_a = set(neighbors_a)
        set_b = set(neighbors_b)
        union = len(set_a | set_b)
        if union == 0:
            return 0.0
        return len(set_a & set_b) / union
    
    def get_neighbors(self, comm_graph: nx.Graph, community_id: int) -> List[int]:
        """Get communication neighbors for a community."""
        if community_id not in comm_graph:
            return []
        return list(comm_graph.neighbors(community_id))
    
    def should_communicate(self, comm_graph: nx.Graph, src: int, dst: int) -> bool:
        """Check if two communities should communicate."""
        return comm_graph.has_edge(src, dst)

import networkx as nx
from collections import defaultdict
import math

def calculate_gamma(G, nodes):
    """
    Calculates the community closedness/clustering intensity gamma(C).
    
    Formula:
        gamma(C) = (1 / |Vc|) * Sum_{u in C} (|N(u) intersect C| / |N(u)|)
        
    Meaning:
        Average proportion of neighbors that remain inside the community.
        
    :param G: NetworkX graph
    :param nodes: List or set of nodes in the community
    :return: gamma value (float)
    """
    nodes_set = set(nodes)
    if not nodes_set:
        return 0.0
    
    total_ratio = 0.0
    valid_nodes_count = 0
    
    for u in nodes_set:
        # Get neighbors (successors if directed, neighbors if undirected)
        if G.is_directed():
            neighbors = list(G.successors(u))
        else:
            neighbors = list(G.neighbors(u))
            
        k_tot = len(neighbors)
        
        if k_tot == 0:
            # Isolated node: convention could be 0 or 1.
            # If a node has no neighbors, it has no "external" neighbors either.
            # But formula denominator is 0. We skip or treat as 0.
            continue
            
        k_in = sum(1 for v in neighbors if v in nodes_set)
        
        total_ratio += (k_in / k_tot)
        valid_nodes_count += 1
        
    # If all nodes are isolated, gamma is 0?
    if len(nodes_set) == 0:
        return 0.0
        
    # The formula says 1/|Vc|, summing over all u in C.
    # Even if k_tot is 0 for some nodes, they are part of Vc.
    # Assuming nodes with degree 0 contribute 0 to the sum.
    return total_ratio / len(nodes_set)

def calculate_gamma_reduction_gain(G, nodes_i, nodes_j):
    """
    Calculates the gamma-reduction gain:
        Delta_gamma_gain(Ci, Cj) = gamma(Ci) - gamma(Ci union Cj)
        
    :return: gain value (float)
    """
    gamma_i = calculate_gamma(G, nodes_i)
    
    nodes_union = list(set(nodes_i).union(set(nodes_j)))
    gamma_union = calculate_gamma(G, nodes_union)
    
    return gamma_i - gamma_union

def calculate_structural_closeness(G, nodes_i, nodes_j):
    """
    Calculates Structural Closeness based on Formula (13).
    
    closeness(Ci, Cj) = 
        Sum_{(u,v) | u in Ci, v in Cj} [ w(u,v) * Sum_{x in N_out(v) inter Cj} w(v,x) ]
      + Sum_{(u,v) | u in Cj, v in Ci} [ w(u,v) * Sum_{x in N_out(v) inter Ci} w(v,x) ]
      
    Meaning:
        Measures connection strength weighted by how "core" (embedded) the target node is in its own community.
    """
    nodes_i_set = set(nodes_i)
    nodes_j_set = set(nodes_j)
    
    score = 0.0
    
    # Term 1: Edges from Ci to Cj
    # Note: edge_boundary returns (u, v, d) where u in set1, v in set2
    edges_ij = list(nx.edge_boundary(G, nodes_i_set, nodes_j_set, data='weight', default=1.0))
    
    for u, v, w_uv in edges_ij:
        # u in Ci, v in Cj
        # Inner sum: Sum_{x in N_out(v) inter Cj} w(v,x)
        # This is the weighted in-degree (or out-degree? N_out) of v within Cj
        if G.is_directed():
            v_neighbors = list(G.successors(v))
        else:
            v_neighbors = list(G.neighbors(v))
            
        term_v = 0.0
        for x in v_neighbors:
            if x in nodes_j_set:
                w_vx = G[v][x].get('weight', 1.0)
                term_v += w_vx
        
        score += w_uv * term_v
        
    # Term 2: Edges from Cj to Ci
    edges_ji = list(nx.edge_boundary(G, nodes_j_set, nodes_i_set, data='weight', default=1.0))
    
    for u, v, w_uv in edges_ji:
        # u in Cj, v in Ci
        # Inner sum: Sum_{x in N_out(v) inter Ci} w(v,x)
        if G.is_directed():
            v_neighbors = list(G.successors(v))
        else:
            v_neighbors = list(G.neighbors(v))
            
        term_v = 0.0
        for x in v_neighbors:
            if x in nodes_i_set:
                w_vx = G[v][x].get('weight', 1.0)
                term_v += w_vx
                
        score += w_uv * term_v
        
    return score

def calculate_merge_score(G, nodes_i, nodes_j, lambda_param=1.0, alpha_param=1.0):
    """
    Calculates the final Gamma-Reduction Merge Score.
    
    MergeScore(Ci, Cj) = StructuralCloseness(Ci, Cj) * (1 + lambda * Delta_gamma_gain(Ci, Cj))^alpha
    
    Note: Since Delta_gamma is asymmetric (gamma(Ci) - gamma(Union)), 
    we must decide if we use i->j or j->i or average.
    However, usually merge identification iterates candidates (i) and finds best partner (j).
    So we calculate from the perspective of candidate i.
    """
    struct_closeness = calculate_structural_closeness(G, nodes_i, nodes_j)
    gamma_gain = calculate_gamma_reduction_gain(G, nodes_i, nodes_j)
    
    # If gamma_gain is negative, (1 + lambda * negative) might be < 0.
    # Usually we clamp or ensure base is positive?
    # Or maybe it's meant to penalize.
    # Assuming the formula works as intended.
    
    base = 1 + lambda_param * gamma_gain
    if base < 0:
        base = 0 # Prevent complex numbers or negative scores if not desired
        
    return struct_closeness * (base ** alpha_param)

def identify_merge_groups_gamma(G, community_nodes_map, candidates, lambda_param=0.8, alpha_param=1.5):
    """
    Identifies merge groups using the Gamma-Reduction Merge Score strategy.
    
    :param G: NetworkX graph
    :param community_nodes_map: Dict {cid: [nodes]}
    :param candidates: List of community IDs that are candidates for merging
    :param lambda_param: Weight for gamma reduction (0.5-1.0 recommended)
    :param alpha_param: Nonlinear amplification factor (1.0-2.0 recommended)
    :return: List of tuples (groups to merge)
    """
    if not candidates:
        return []

    all_ids = list(community_nodes_map.keys())
    max_connection_partner = {cid: -1 for cid in all_ids}
    
    # Find best partner for each candidate
    for i in candidates:
        max_score = -float('inf')
        best_partner = -1
        
        nodes_i = community_nodes_map[i]
        
        for j in all_ids:
            if i == j: continue
            
            nodes_j = community_nodes_map[j]
            
            # Calculate Score
            score = calculate_merge_score(G, nodes_i, nodes_j, lambda_param, alpha_param)
            
            if score > max_score:
                max_score = score
                best_partner = j
        
        # Threshold check? (Optional, if score is too low maybe don't merge)
        # For now, we take the max.
        if max_score > 0:
             max_connection_partner[i] = best_partner
        else:
             max_connection_partner[i] = -1

    # Resolve Merge Groups (Connected Components logic)
    adj = defaultdict(set)
    for i in candidates:
        target = max_connection_partner[i]
        if target != -1:
            adj[i].add(target)
            adj[target].add(i)
            
    visited = set()
    groups = []
    
    for start_node in candidates:
        if start_node not in visited:
            component = set()
            stack = [start_node]
            visited.add(start_node)
            while stack:
                u = stack.pop()
                component.add(u)
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        stack.append(v)
            
            if len(component) > 1:
                groups.append(tuple(component))
                
    return groups

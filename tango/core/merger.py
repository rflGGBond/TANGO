import copy
import networkx as nx
from collections import defaultdict
import heapq
from .evaluator import DPADVEvaluator

def calculate_connection_strength(G, nodes_i, nodes_j):
    """
    Calculates the connection strength (score) between two communities based on boundary edges.
    Extracted from merge_communities logic.
    """
    score = 0
    # Calculate connection strength (boundary edges * weight)
    # Note: original logic had complex subG_list neighbor weight lookups.
    # Simplified here to just sum of edge weights between groups for robustness and speed,
    # or strictly follow original if subG is available.
    # Original logic:
    # merge_score[i, j] += (one_score * G[edge[0]][edge[1]]['weight'])
    # where one_score was sum of weights of neighbors in subG.
    
    # For HMACE Env usage, we use direct edge weight sum as a robust proxy.
    try:
        edges = list(nx.edge_boundary(G, nodes_i, nodes_j, data='weight', default=1.0))
        for u, v, w in edges:
            score += w
            
        # If we want to strictly mimic the "neighbor weight * edge weight" logic:
        # That logic seems to imply 2-hop or influence weight. 
        # For now, we return the direct connection strength.
        return score
    except Exception:
        return 0.0

def identify_merge_groups(G, community_nodes_map, candidates):
    """
    Identifies which communities should be merged based on connection strength.
    
    :param G: NetworkX graph
    :param community_nodes_map: Dict {community_id: [nodes]}
    :param candidates: List of community IDs that are flagged for merging (stagnant)
    :return: List of tuples, where each tuple contains community IDs to be merged.
    """
    if not candidates:
        return []

    # 1. Initialize Merge Flags
    # -1: Needs merge (candidate)
    # -2: Stable (not candidate)
    # This mimics the original PCMCC merge array logic
    all_ids = list(community_nodes_map.keys())
    merge_flags = {cid: -2 for cid in all_ids}
    for cid in candidates:
        merge_flags[cid] = -1
        
    # 2. Calculate Connection Strengths & Determine Pairings
    # We iterate through candidates to find their best partner
    
    # Store max connection for each community
    max_connection_partner = {cid: -1 for cid in all_ids}
    
    # Cache for connection scores to avoid re-calculation
    # Key: (min(i,j), max(i,j)) -> score
    score_cache = {}

    def get_score(cid1, cid2):
        key = tuple(sorted((cid1, cid2)))
        if key in score_cache:
            return score_cache[key]
        
        # PCMCC Metric: sum(internal_weight * boundary_weight)
        # We need the subgraph of each community to get internal weights.
        # Since passing subgraphs is heavy, we calculate on the fly using G.
        # Ideally, we should use the exact formula.
        
        score = 0
        nodes1 = set(community_nodes_map[cid1])
        nodes2 = set(community_nodes_map[cid2])
        
        # Find boundary edges
        try:
            boundary_edges = list(nx.edge_boundary(G, nodes1, nodes2))
            
            for u, v in boundary_edges:
                # u in nodes1, v in nodes2 (or vice versa depending on direction, edge_boundary handles it)
                # But G is directed or undirected? 
                # PCMCC logic:
                # for edge in boundary(i, j):
                #   one_score = sum(weight(v, x) for x in neighbors(v) inside j)
                #   score += one_score * weight(u, v)
                
                # Let's assume u is in cid1, v in cid2.
                # We need contribution from cid2 (neighbors of v in cid2)
                # AND contribution from cid1 (neighbors of u in cid1)
                
                w_uv = G[u][v]['weight']
                
                # Part 1: Contribution from cid2 side
                # Sum of weights from v to its neighbors in cid2
                weight_in_2 = 0
                if v in nodes2: # Should be true
                    for neighbor in G.neighbors(v):
                        if neighbor in nodes2:
                             weight_in_2 += G[v][neighbor]['weight']
                
                score += weight_in_2 * w_uv
                
                # Part 2: Contribution from cid1 side
                # Sum of weights from u to its neighbors in cid1
                weight_in_1 = 0
                if u in nodes1: # Should be true
                    for neighbor in G.neighbors(u):
                        if neighbor in nodes1:
                            weight_in_1 += G[u][neighbor]['weight']
                            
                score += weight_in_1 * w_uv
                
        except Exception:
            score = 0
            
        score_cache[key] = score
        return score

    # Find best partner for each candidate
    for i in candidates:
        max_score = -1
        best_partner = -1
        
        for j in all_ids:
            if i == j: continue
            
            s = get_score(i, j)
            if s > max_score:
                max_score = s
                best_partner = j
        
        max_connection_partner[i] = best_partner

    # 3. Resolve Merge Flags (Transitive Closure Logic from PCMCC)
    # Original logic:
    # for i in sorted_by_size:
    #   if merge[i] == -1:
    #     target = best_partner[i]
    #     if merge[target] is virgin(-2) or needs_merge(-1):
    #        merge[i] = i
    #        merge[target] = i
    #     else:
    #        merge[i] = merge[target]
    
    # We simplify this to: Build a graph where edges are (i, best_partner[i])
    # and find connected components.
    # But we must respect the "direction" of merge to some extent?
    # Actually, connected components is the robust way to say "these guys merge together".
    
    # Build adjacency for components
    adj = defaultdict(set)
    for i in candidates:
        target = max_connection_partner[i]
        if target != -1:
            adj[i].add(target)
            adj[target].add(i)
            
    # Find components
    visited = set()
    groups = []
    
    # Only start traversal from candidates (nodes that initiated the merge)
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

def merge_communities(merge_flags, community_list, community_k, population, effect,
                      com_res, Ni, G, subG_list, SN, fitness_space, hop, s_t_l, curT,
                      com_gen_acc, com_ben, P_score, gama, search_space):
    """
    Refactored from mergeCommunity_12.
    Executes the community merging process based on merge_flags.
    """
    
    # Logic to determine merge pairs (transitive closure of merge flags)
    if len(community_list) > 2:
        lengths = [len(sublist) for sublist in community_list]
        # Sort by size descending
        lengths_index = [i[0] for i in sorted(enumerate(lengths), key=lambda x: x[1], reverse=True)]
        max_connection_index = [-1 for _ in range(len(community_list))]

        merge_score = defaultdict(lambda: -1)

        for i in range(len(community_list)):
            if merge_flags[i] == -1:
                max_connection_i = 0
                max_connection_index_i = 0
                for j in range(len(community_list)):
                    if i != j:
                        if merge_score[i, j] == -1:
                            merge_score[i, j] = 0
                            merge_score[j, i] = 0
                            
                            # Calculate connection strength (boundary edges * weight)
                            for edge in list(nx.edge_boundary(G, community_list[i], community_list[j])):
                                one_score = 0
                                for v in subG_list[j].neighbors(edge[1]):
                                    one_score += subG_list[j][edge[1]][v]['weight']
                                merge_score[i, j] += (one_score * G[edge[0]][edge[1]]['weight'])
                                merge_score[j, i] += (one_score * G[edge[0]][edge[1]]['weight'])

                                one_score = 0
                                for v in subG_list[i].neighbors(edge[0]):
                                    one_score += subG_list[i][edge[0]][v]['weight']
                                merge_score[i, j] += (one_score * G[edge[0]][edge[1]]['weight'])
                                merge_score[j, i] += (one_score * G[edge[0]][edge[1]]['weight'])

                        if max_connection_i <= merge_score[i, j]:
                            max_connection_i = merge_score[i, j]
                            max_connection_index_i = j

                max_connection_index[i] = max_connection_index_i

        for i in lengths_index:
            if merge_flags[i] == -1:
                if (merge_flags[max_connection_index[i]] == -2) or (merge_flags[max_connection_index[i]] == -1):
                    merge_flags[i] = i
                    merge_flags[max_connection_index[i]] = i
                else:
                    merge_flags[i] = merge_flags[max_connection_index[i]]

    elif len(community_list) == 2:
        if merge_flags[0] == -1 and merge_flags[1] == -2:
            merge_flags[0] = 0
            merge_flags[1] = 0
        elif merge_flags[0] == -2 and merge_flags[1] == -1:
            merge_flags[0] = 1
            merge_flags[1] = 1
        elif merge_flags[0] == -1 and merge_flags[1] == -1:
            merge_flags[0] = 0
            merge_flags[1] = 0

    # Group communities to be merged
    my_dict = {}
    for i, value in enumerate(merge_flags):
        if value >= 0:
            if value not in my_dict:
                my_dict[value] = [i]
            else:
                my_dict[value].append(i)

    sorted_dict = sorted(my_dict.items(), key=lambda x: len(x[1]), reverse=True)
    to_be_merged = [x[1] for x in sorted_dict if len(x[1]) > 1]

    for i in range(len(community_list)):
        rs = True
        for row in to_be_merged:
            if i in row:
                rs = False
        if rs:
            to_be_merged.append([i])

    # Re-organize data structures
    new_com_res = []
    for i in range(len(to_be_merged)):
        new_com_res.append(0)
        for j in to_be_merged[i]:
            new_com_res[i] += com_res[j]

    new_community_k = []
    new_community_list = []
    
    # Initialize new islands (population)
    # Note: Structure is [Community][Subpop][Individual]
    # This logic assumes 'population' is a list of lists of lists
    
    new_islands = [[[] for j in range(new_com_res[i])] for i in range(len(to_be_merged))]
    new_islands_effect = [[[] for j in range(new_com_res[i])] for i in range(len(to_be_merged))]

    for i in range(len(to_be_merged)):
        new_community_k.append(0)
        new_community_list.append([])

        for j in to_be_merged[i]:
            new_community_k[i] += community_k[j]
            new_community_list[i] += community_list[j]

    # Recalculate helper sets for new communities
    com_and_sea = []
    com_and_fs = []
    com_or_sn = []
    com_gs = []
    gama_com = []
    
    for i in range(len(new_community_list)):
        com_and_sea.append(list(set(search_space).intersection(set(new_community_list[i]))))
        com_and_fs.append(list(set(new_community_list[i]).intersection(set(fitness_space))))
        com_or_sn.append(list(set(new_community_list[i] + SN)))
        
        tempsubGi = G.subgraph(com_or_sn[i])
        subGi = nx.Graph(tempsubGi.edges(data=True))
        subGi.add_nodes_from(com_or_sn[i])
        com_gs.append(subGi.copy())
        
        gama_com.append(
            heapq.nlargest(min(int(round(gama * new_community_k[i])), len(com_and_sea[i])), com_and_sea[i],
                           key=lambda x: P_score[x]))

    # Merge populations
    for i in range(len(to_be_merged)):
        temp_islands_i = [[[] for N in range(Ni)] for J in range(new_com_res[i])]
        for J in range(new_com_res[i]):
            for N in range(Ni):
                for j in to_be_merged[i]:
                    # Distribute old individuals to new structure
                    # This logic attempts to merge populations from multiple source communities
                    # into the new larger community's subpopulations
                    # Original logic is complex here, keeping it as is:
                    # It seems to flatten old populations and redistribute
                    temp_islands_i[J][N] += population[j][J % com_res[j]][N]

        for J in range(new_com_res[i]):
            for N in range(Ni):
                new_islands[i][J].append(temp_islands_i[J][N])

    # Recalculate effects
    for i in range(len(to_be_merged)):
        if len(to_be_merged[i]) == 1:
            for J in range(new_com_res[i]):
                for N in range(Ni):
                    new_islands_effect[i][J].append(effect[to_be_merged[i][0]][J][N])
        else:
            for J in range(new_com_res[i]):
                for N in range(Ni):
                    new_islands_effect[i][J].append(
                        DPADVEvaluator.calculate_fitness(new_islands[i][J][N], com_gs[i], SN,
                                    com_and_fs[i], hop))

    # Calculate min effects for history
    minE = [0 for i in range(len(to_be_merged))]
    for i in range(len(to_be_merged)):
        if new_islands_effect[i] and new_islands_effect[i][0]:
             minE_i = new_islands_effect[i][0][0]
             for j in range(new_com_res[i]):
                 if min(new_islands_effect[i][j]) < minE_i:
                     minE_i = min(new_islands_effect[i][j])
             minE[i] = minE_i

    new_com_gen_acc = [-1 for i in range(len(to_be_merged))]
    new_com_ben = [-1 for i in range(len(to_be_merged))]

    # Update history records (s_t_l)
    for i in range(len(to_be_merged)):
        if len(to_be_merged[i]) == 1:
            new_com_gen_acc[i] = com_gen_acc[to_be_merged[i][0]]
            new_com_ben[i] = com_ben[to_be_merged[i][0]]
            
            # Map old history to new index
            # Note: s_t_l uses (index, time, total_coms) as key. 
            # This is tricky because total_coms changes.
            # We assume the caller handles the re-keying or we follow original logic closely.
            s_t_l[i, curT, len(to_be_merged)] = (minE[i], com_gen_acc[to_be_merged[i][0]])
            # Copy benchmark record
            s_t_l[i, new_com_ben[i], len(to_be_merged)] = copy.deepcopy(
                s_t_l[to_be_merged[i][0], com_ben[to_be_merged[i][0]], len(community_list)])

        else:
            new_com_gen_acc[i] = com_gen_acc[to_be_merged[i][0]]
            new_com_ben[i] = curT # Reset benchmark time for new merged community

            for j in to_be_merged[i]:
                if new_com_gen_acc[i] > com_gen_acc[j]:
                    new_com_gen_acc[i] = com_gen_acc[j]

            s_t_l[i, curT, len(to_be_merged)] = (minE[i], new_com_gen_acc[i])

    return new_islands, new_islands_effect, new_community_list, new_community_k, \
           s_t_l, new_com_gen_acc, new_com_ben, com_and_sea, com_and_fs, com_or_sn, com_gs, gama_com, new_com_res

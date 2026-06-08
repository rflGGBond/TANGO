import re
import copy
import igraph as ig
import leidenalg
import networkx as nx

def detect_communities(G, num_communities):
    """
    Divides the graph G into num_communities using Leiden algorithm.
    Refactored from communityDivision_1.
    """
    c_G = G.copy()
    is_directed = G.is_directed()
    c_g = copy.deepcopy(ig.Graph.TupleList(list(c_G.edges(data='weight')), directed=is_directed, edge_attrs=['weight']))
    c_part = leidenalg.find_partition(c_g, leidenalg.ModularityVertexPartition, weights=c_g.es['weight'],
                                      n_iterations=-1)
    
    print(f"Modularity: {c_part.modularity}\n")

    rs_part = []
    # Using regex to parse the partition output string (as in original code)
    # Note: This is a bit brittle, but keeping original logic for consistency unless requested otherwise.
    # Ideally, we should iterate over c_part directly if possible, but c_part structure depends on library version.
    # Let's try to trust the original regex logic for now.
    
    pattern1 = re.compile(r"(?<=])[^][]+(?=\n\[)")
    matches = pattern1.findall(str(c_part) + "\n[")
    for match1 in matches:
        pattern2 = r"\d+"
        numbers = [int(match2) for match2 in re.findall(pattern2, match1)]
        rs_part.append(numbers)

    # Adjust number of communities
    while len(rs_part) != num_communities:
        if len(rs_part) > num_communities:
            lengths = [len(lst) for lst in rs_part]
            min_indices = sorted(range(len(lengths)), key=lambda k: lengths[k])[:2]
            rs_part[min_indices[1]].extend(rs_part[min_indices[0]])
            del rs_part[min_indices[0]]
        if len(rs_part) < num_communities:
            lengths = [len(lst) for lst in rs_part]
            max_indices = sorted(range(len(lengths)), key=lambda k: lengths[k])[-1]
            a_G = G.subgraph(rs_part[max_indices]).copy()
            a_g = copy.deepcopy(
                ig.Graph.TupleList(list(a_G.edges(data='weight')), directed=is_directed, edge_attrs=['weight']))
            a_part = leidenalg.find_partition(a_g, leidenalg.ModularityVertexPartition, weights=a_g.es['weight'],
                                              n_iterations=-1)
            a_rs_part = []
            pattern_a = re.compile(r"(?<=])[^][]+(?=\n\[)")
            matches_a = pattern_a.findall(str(a_part) + "\n[")
            for match1_a in matches_a:
                pattern2_a = r"\d+"
                numbers_a = [int(match2_a) for match2_a in re.findall(pattern2_a, match1_a)]
                a_rs_part.append(numbers_a)
            
            # Fix infinite loop: Force split if Leiden returns < 2 partitions
            if len(a_rs_part) < 2:
                print(f"Warning: Leiden could not split community. Forcing random split.")
                target_nodes = rs_part[max_indices]
                mid = len(target_nodes) // 2
                if mid > 0:
                    a_rs_part = [target_nodes[:mid], target_nodes[mid:]]

            del rs_part[max_indices]
            rs_part[max_indices:max_indices] = copy.deepcopy(a_rs_part)

    return rs_part

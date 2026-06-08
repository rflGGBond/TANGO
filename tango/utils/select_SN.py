import networkx as nx

def select_SN(graph_name, SN_size, is_directed=False):
    file_path = f"../graph/{graph_name}.txt"

    G = nx.DiGraph() if is_directed else nx.Graph()

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            u, v = int(parts[0]), int(parts[1])
            w = float(parts[2])
            G.add_edge(u, v, weight=w)
    
    # 按 degree 从大到小排序
    # User requested Degree (Total Degree) for both Directed and Undirected graphs.
    # In NetworkX, G.degree() returns (in_degree + out_degree) for DiGraph.
    degree_list = sorted(G.degree(), key=lambda x: x[1], reverse=True)

    # 取前 50 个节点
    SN = [node for node, deg in degree_list[:SN_size]]
    
    return SN

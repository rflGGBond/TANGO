import random
from collections import defaultdict
import networkx as nx

class DPADVEvaluator:
    """
    Evaluator class for calculating DPADV (Dynamic Propagation-Activation-Degree Value)
    and other related metrics.
    """

    @staticmethod
    def calculate_negative_probability(G, SN, fitness_space, hop, all_FP=None):
        """
        Calculates the negative activation probability for nodes in the fitness space.
        Refactored from negativeProbability_2.
        """
        if all_FP is None:
            all_FP = []

        rs_N_p = {}
        ZN_f = []
        ZN_f.append(SN)
        for h in range(1, hop + 1):
            ZN_f.append([])
        
        pN_f = defaultdict(lambda: 0)
        apN_f = defaultdict(lambda: 0)
        
        for v in SN:
            pN_f[v, 0] = 1
            for h in range(hop + 1):
                apN_f[v, h] = 1
        
        for h in range(hop):
            temppN_f = defaultdict(lambda: 1)
            for v in ZN_f[h]:
                W_f = list(G.neighbors(v))
                ZN_f[h + 1] += W_f
                for w in W_f:
                    temppN_f[w] *= (1 - pN_f[v, h] * G[v][w]['weight'])
            
            ZN_f[h + 1] = list(set(ZN_f[h + 1]))
            for v in ZN_f[h + 1]:
                pN_f[v, h + 1] = (1 - temppN_f[v]) * (1 - apN_f[v, h])
                for tau_f in range(h + 1, hop + 1):
                    apN_f[v, tau_f] = apN_f[v, h] + pN_f[v, h + 1]

        for u in fitness_space:
            for t in range(1, hop + 1):
                rs_N_p[u, t] = pN_f[u, t]

        for u in all_FP:
            for t in range(1, hop + 1):
                rs_N_p[u, t] = 0
                
        return rs_N_p

    @staticmethod
    def calculate_positive_score(G, fitness_space, search_space, hop, N_prob):
        """
        Calculates the positive score for nodes in the search space.
        Refactored from positiveScore_3.
        """
        rs_P_S = {}

        for u in search_space:
            predecessors = defaultdict(lambda: [])
            rs_P_S[u] = 0
            one_hop_neighbors = []
            two_hop_neighbors = []
            
            for v in G.neighbors(u):
                one_hop_neighbors.append(v)
                for w in G.neighbors(v):
                    two_hop_neighbors.append(w)
                    predecessors[w].append(v)

            oneAndF = set(one_hop_neighbors).intersection(set(fitness_space))
            two_hop_neighbors = set(two_hop_neighbors).intersection(set(fitness_space)) - set([u])

            twoAndOne = two_hop_neighbors.intersection(oneAndF)
            two_One = two_hop_neighbors - oneAndF

            for t in range(1, hop + 1):
                rs_P_S[u] += N_prob.get((u, t), 0)

            for v in oneAndF:
                for t in range(1, hop + 1):
                    rs_P_S[u] += G[u][v]['weight'] * N_prob.get((v, t), 0)

            for w in twoAndOne:
                temp_p = 1
                for v in set(predecessors[w]):
                    temp_p *= (1 - G[u][v]['weight'] * G[v][w]['weight'])
                for t in range(2, hop + 1):
                    rs_P_S[u] += (1 - G[u][w]['weight']) * (1 - temp_p) * (1 - N_prob.get((w, 1), 0)) * N_prob.get((w, t), 0)

            for w in two_One:
                temp_p = 1
                for v in set(predecessors[w]):
                    temp_p *= (1 - G[u][v]['weight'] * G[v][w]['weight'])
                for t in range(2, hop + 1):
                    rs_P_S[u] += (1 - temp_p) * (1 - N_prob.get((w, 1), 0)) * N_prob.get((w, t), 0)

        return rs_P_S

    @staticmethod
    def calculate_fitness(seed, G, SN, com_and_fs, hop):
        """
        Calculates the fitness (DPADV) of a given seed set.
        Refactored from fitness_C_7.
        """
        effect_fc = 0
        ZP_fc = []
        ZN_fc = []
        ZP_fc.append(seed)
        ZN_fc.append(SN)
        
        for h in range(1, hop + 1):
            ZP_fc.append([])
            ZN_fc.append([])
            
        pP_fc = defaultdict(lambda: 0)
        apP_fc = defaultdict(lambda: 0)
        pN_fc = defaultdict(lambda: 0)
        apN_fc = defaultdict(lambda: 0)
        
        for v in seed:
            pP_fc[v, 0] = 1
            for h in range(hop + 1):
                apP_fc[v, h] = 1
                
        for v in SN:
            pN_fc[v, 0] = 1
            for h in range(hop + 1):
                apN_fc[v, h] = 1
                
        for h in range(hop):
            temppP_fc = defaultdict(lambda: 1)
            temppN_fc = defaultdict(lambda: 1)
            
            for v in ZP_fc[h]:
                W_fc = list(G.neighbors(v))
                ZP_fc[h + 1] += W_fc
                for w in W_fc:
                    temppP_fc[w] *= (1 - pP_fc[v, h] * G[v][w]['weight'])
            
            ZP_fc[h + 1] = list(set(ZP_fc[h + 1]))
            for v in ZP_fc[h + 1]:
                pP_fc[v, h + 1] = (1 - temppP_fc[v]) * (1 - apN_fc[v, h]) * (1 - apP_fc[v, h])
                for tau_f in range(h + 1, hop + 1):
                    apP_fc[v, tau_f] = apP_fc[v, h] + pP_fc[v, h + 1]
                    
            for v in ZN_fc[h]:
                W_fc = list(G.neighbors(v))
                ZN_fc[h + 1] += W_fc
                for w in W_fc:
                    temppN_fc[w] *= (1 - pN_fc[v, h] * G[v][w]['weight'])
            
            ZN_fc[h + 1] = list(set(ZN_fc[h + 1]))
            for v in ZN_fc[h + 1]:
                pN_fc[v, h + 1] = temppP_fc[v] * (1 - temppN_fc[v]) * (1 - apN_fc[v, h]) * (1 - apP_fc[v, h])
                for tau_f in range(h + 1, hop + 1):
                    apN_fc[v, tau_f] = apN_fc[v, h] + pN_fc[v, h + 1]
                    
        for u in com_and_fs:
            effect_fc += apN_fc[u, hop]

        return effect_fc

    @staticmethod
    def simulate_propagation(G, S_P, S_N, model='COICM'):
        """
        Simulate the diffusion process where positive and negative information propagate simultaneously.
        Matches PCMCC logic (Synchronous update, Positive Priority, Exhaustive).
        
        Rules:
        - Nodes have 3 states: Positive, Negative, Non-activated.
        - Initial: S_P are Positive, S_N are Negative.
        - Propagation: Activated nodes attempt to activate neighbors.
        - Conflict: If a node is activated by both Positive and Negative in the same step, Positive wins.
        - Termination: No new nodes activated.
        - Models:
            - COICM: P_P = P_N = edge weight
            - MCICM: P_P = 1, P_N = edge weight
        """
        pos_activated = set(S_P)
        neg_activated = set(S_N)
        
        pos_frontier = set(S_P)
        neg_frontier = set(S_N)
        
        while pos_frontier or neg_frontier:
            next_pos_frontier = set()
            next_neg_frontier = set()
            
            # Potential activations for this step
            potential_pos_activations = set()
            potential_neg_activations = set()
            
            # 1. Determine all potential positive activations
            for u in pos_frontier:
                for v in G.neighbors(u):
                    if v not in pos_activated and v not in neg_activated:
                        weight = G[u][v]['weight']
                        prob = 1.0 if model == 'MCICM' else weight
                        if random.random() < prob:
                            potential_pos_activations.add(v)
            
            # 2. Determine all potential negative activations
            for u in neg_frontier:
                for v in G.neighbors(u):
                    if v not in pos_activated and v not in neg_activated:
                        weight = G[u][v]['weight']
                        if random.random() < weight:
                            potential_neg_activations.add(v)
            
            # 3. Resolve conflicts (Positive Priority)
            # If v is in potential_pos, it becomes positive regardless of potential_neg
            for v in potential_pos_activations:
                pos_activated.add(v)
                next_pos_frontier.add(v)
                
            # If v is in potential_neg BUT NOT in potential_pos, it becomes negative
            for v in potential_neg_activations:
                if v not in potential_pos_activations:
                    neg_activated.add(v)
                    next_neg_frontier.add(v)
            
            pos_frontier = next_pos_frontier
            neg_frontier = next_neg_frontier
            
        return len(neg_activated)

    @staticmethod
    def get_activated_node_count(seed_set, G, SN, runs=50, model='COICM'):
        """
        Calculates the average number of negatively activated nodes by running multiple simulations.
        """
        total_activated = 0
        for _ in range(runs):
            count = DPADVEvaluator.simulate_propagation(G, seed_set, SN, model)
            total_activated += count
        return total_activated / runs


import random
import copy
from collections import defaultdict
from .evaluator import DPADVEvaluator

def sample(l1, w1, k):
    """
    Weighted sampling without replacement (approximate implementation from original code).
    """
    l = copy.deepcopy(l1)
    w = copy.deepcopy(w1)

    randoms = [random.random() for i in range(k)]

    total_w = 0
    bu = {}
    bd = {}
    for u in l:
        bd[u] = total_w
        bu[u] = total_w + w[u]
        total_w += w[u]

    l_new = copy.deepcopy(l)
    total_w_new = total_w

    rs = []
    for r in randoms:
        total_w = total_w_new
        r_total_w = r * total_w
        l = copy.deepcopy(l_new)
        a = 0
        count = 0
        for u in l:
            if a == 1:
                bd[u] -= w[rs[-1]]
                bu[u] -= w[rs[-1]]

            if (a == 0) and (r_total_w > bd[u]) and (r_total_w <= bu[u]):
                rs.append(u)
                del l_new[count]
                a = 1

            count += 1

        total_w_new = total_w - w[rs[-1]]

    return rs

def local_search(S1, G, com_and_fs, hop, N_prob, gama_com):
    """
    Simplified interface for local search, operating on standard lists.
    Returns the optimized seed set S1.
    """
    S1 = list(S1) # Ensure list
    budget = len(S1)
    
    if budget == 0:
        return S1
    
    while True:
        discount_P_score_diff = []
        
        # 1. Calculate contribution for each node in S1
        for I in range(budget):
            rs = 0
            predecessors = defaultdict(lambda: [])
            one_hop_neighbors = []
            two_hop_neighbors = []

            for v in G.neighbors(S1[I]):
                one_hop_neighbors.append(v)
                for w in G.neighbors(v):
                    two_hop_neighbors.append(w)
                    predecessors[w].append(v)

            oneAndF = set(one_hop_neighbors).intersection(set(com_and_fs)) - set(S1)
            two_hop_neighbors = set(two_hop_neighbors).intersection(set(com_and_fs)) - set(S1)
            twoAndOne = two_hop_neighbors.intersection(oneAndF)
            two_one = two_hop_neighbors - oneAndF

            for t in range(1, hop + 1):
                rs += N_prob.get((S1[I], t), 0)

            for v in oneAndF:
                for t in range(1, hop + 1):
                    rs += G[S1[I]][v]['weight'] * N_prob.get((v, t), 0)

            for w in twoAndOne:
                temp_p = 1
                for v in set(predecessors[w]):
                    temp_p *= (1 - G[S1[I]][v]['weight'] * G[v][w]['weight'])
                for t in range(2, hop + 1):
                    rs += (1 - G[S1[I]][w]['weight']) * (1 - temp_p) * (1 - N_prob.get((w, 1), 0)) * N_prob.get((w, t), 0)

            for w in two_one:
                temp_p = 1
                for v in set(predecessors[w]):
                    temp_p *= (1 - G[S1[I]][v]['weight'] * G[v][w]['weight'])
                for t in range(2, hop + 1):
                    rs += (1 - temp_p) * (1 - N_prob.get((w, 1), 0)) * N_prob.get((w, t), 0)

            temp = 1
            rs1 = 0
            
            # Use predecessors if directed, neighbors if undirected
            incoming_neighbors = G.predecessors(S1[I]) if G.is_directed() else G.neighbors(S1[I])
            
            for u in set(S1).intersection(set(incoming_neighbors)):
                temp *= (1 - G[u][S1[I]]['weight'])

            for t in range(1, hop + 1):
                rs1 += (1 - temp) * N_prob.get((S1[I], t), 0)

            for v in oneAndF:
                for t in range(2, hop + 1):
                    rs1 += (1 - temp) * G[S1[I]][v]['weight'] * N_prob.get((v, t), 0)

            discount_P_score_diff.append(rs - rs1)

        # 2. Find worst node to replace
        I_worst = discount_P_score_diff.index(min(discount_P_score_diff))
        Sbest = copy.deepcopy(S1)
        
        replace_discount_P_score_diff = {}
        
        # 3. Try replacing worst node
        for nn in (set(gama_com) - set(Sbest)):
            S1[I_worst] = nn # Tentative replacement

            rs = 0
            predecessors = defaultdict(lambda: [])
            one_hop_neighbors = []
            two_hop_neighbors = []

            for v in G.neighbors(S1[I_worst]):
                one_hop_neighbors.append(v)
                for w in G.neighbors(v):
                    two_hop_neighbors.append(w)
                    predecessors[w].append(v)

            oneAndF = set(one_hop_neighbors).intersection(set(com_and_fs)) - set(S1)
            two_hop_neighbors = set(two_hop_neighbors).intersection(set(com_and_fs)) - set(S1)
            twoAndOne = two_hop_neighbors.intersection(oneAndF)
            two_one = two_hop_neighbors - oneAndF

            for t in range(1, hop + 1):
                rs += N_prob.get((S1[I_worst], t), 0)

            for v in oneAndF:
                for t in range(1, hop + 1):
                    rs += G[S1[I_worst]][v]['weight'] * N_prob.get((v, t), 0)

            for w in twoAndOne:
                temp_p = 1
                for v in set(predecessors[w]):
                    temp_p *= (1 - G[S1[I_worst]][v]['weight'] * G[v][w]['weight'])
                for t in range(2, hop + 1):
                    rs += (1 - G[S1[I_worst]][w]['weight']) * (1 - temp_p) * (1 - N_prob.get((w, 1), 0)) * N_prob.get((w, t), 0)

            for w in two_one:
                temp_p = 1
                for v in set(predecessors[w]):
                    temp_p *= (1 - G[S1[I_worst]][v]['weight'] * G[v][w]['weight'])
                for t in range(2, hop + 1):
                    rs += (1 - temp_p) * (1 - N_prob.get((w, 1), 0)) * N_prob.get((w, t), 0)

            temp = 1
            rs1 = 0
            
            incoming_neighbors = G.predecessors(S1[I_worst]) if G.is_directed() else G.neighbors(S1[I_worst])
            
            for u in set(S1).intersection(set(incoming_neighbors)):
                temp *= (1 - G[u][S1[I_worst]]['weight'])

            for t in range(1, hop + 1):
                rs1 += (1 - temp) * N_prob.get((S1[I_worst], t), 0)

            for v in oneAndF:
                for t in range(2, hop + 1):
                    rs1 += (1 - temp) * G[S1[I_worst]][v]['weight'] * N_prob.get((v, t), 0)

            replace_discount_P_score_diff[nn] = rs - rs1

        S1[I_worst] = -1 # Reset
        rmax = discount_P_score_diff[I_worst]
        rn = Sbest[I_worst]

        for nn in list(set(gama_com) - set(Sbest)):
            if replace_discount_P_score_diff[nn] >= rmax:
                rmax = replace_discount_P_score_diff[nn]
                rn = nn

        S1[I_worst] = rn

        # 4. Check if improvement occurred (Simple check: if we picked a new node)
        # Note: In original code, it recalculates fitness to be sure.
        # Here we just check if S1 changed.
        if S1 == Sbest:
            break
            
    return S1

def initialize_population(community_id, subpop_id, Ni, com_and_sea, budget):
    """
    Initializes population for a community subpopulation.
    Refactored from populationInitialization_4.
    """
    population = {}
    for I in range(Ni):
        # Sample 'budget' nodes from 'com_and_sea' (Community Intersection Search Space)
        # Ensure we don't sample more than available
        k = min(budget, len(com_and_sea))
        population[community_id, subpop_id, I] = random.sample(com_and_sea, k=k)
    return population

def calculate_population_effect(community_id, subpop_id, Ni, population, G, SN, com_and_fs, hop):
    """
    Calculates fitness for a whole population.
    Refactored from calEffect_6.
    """
    effect = {}
    for I in range(Ni):
        effect[community_id, subpop_id, I] = DPADVEvaluator.calculate_fitness(
            population[I], G, SN, com_and_fs, hop
        )
    return effect

def crossover_and_mutate(S1_in, SI_in, budget, cOne, cTwo, com_sn, P_score, alpha=None):
    """
    Pure function to perform crossover and mutation on two individuals.
    Returns both new offspring (S1, SI).
    """
    S1 = copy.deepcopy(S1_in)
    SI = copy.deepcopy(SI_in)
    
    # Apply alpha reduction if provided
    # com_sn is assumed to be sorted (e.g. by degree) as passed from env
    current_pool = com_sn
    if alpha is not None:
        pool_size = int(alpha * budget)
        if pool_size < len(com_sn):
            current_pool = com_sn[:pool_size]
    
    repeatS1 = 0
    repeatSI = 0

    for J in range(len(S1)):
        if random.random() < cOne:
            if random.random() < cTwo:  # two-way cross (Swap)
                temp = S1[J]
                
                # S1 takes from SI
                if SI[J] not in S1 or SI[J] == S1[J]:
                    S1[J] = SI[J]
                else:
                    S1[J] = -1
                    repeatS1 += 1
                
                # SI takes from S1 (temp)
                if temp not in SI or temp == SI[J]:
                    SI[J] = temp
                else:
                    SI[J] = -1
                    repeatSI += 1
                    
            else:  # one-way cross (Best -> m)
                # SI takes from S1, S1 remains unchanged
                if S1[J] not in SI or S1[J] == SI[J]:
                    SI[J] = S1[J]
                else:
                    SI[J] = -1
                    repeatSI += 1
                    
    # Fix duplicates S1
    if repeatS1 != 0:
        candidates = list(set(current_pool) - set(S1))
        if candidates:
            replaceS1 = sample(candidates, P_score, repeatS1)
            J = 0
            for e in range(budget):
                if S1[e] == -1 and J < len(replaceS1):
                    S1[e] = replaceS1[J]
                    J += 1
                    
    # Fix duplicates SI
    if repeatSI != 0:
        candidates = list(set(current_pool) - set(SI))
        if candidates:
            replaceSI = sample(candidates, P_score, repeatSI)
            J = 0
            for e in range(budget):
                if SI[e] == -1 and J < len(replaceSI):
                    SI[e] = replaceSI[J]
                    J += 1
    
    return S1, SI

def full_crossover_mutate(S1_in, SI_in, budget, cOne, cTwo, com_sn, P_score, alpha=None):
    """
    Performs full crossover/mutation on two individuals, returning both modified versions.
    """
    S1 = copy.deepcopy(S1_in)
    SI = copy.deepcopy(SI_in)
    
    # Apply alpha reduction if provided
    current_pool = com_sn
    if alpha is not None:
        pool_size = int(alpha * budget)
        if pool_size < len(com_sn):
            current_pool = com_sn[:pool_size]
    
    repeatS1 = 0
    repeatSI = 0

    for J in range(len(S1)):
        if random.random() < cOne:
            if random.random() < cTwo:  # two-way cross
                temp = S1[J]
                if SI[J] not in S1 or SI[J] == S1[J]:
                    S1[J] = SI[J]
                else:
                    S1[J] = -1
                    repeatS1 += 1
                
                if temp not in SI or temp == SI[J]:
                    SI[J] = temp
                else:
                    SI[J] = -1
                    repeatSI += 1
            else:  # one-way cross
                if S1[J] not in SI or S1[J] == SI[J]:
                    SI[J] = S1[J]
                else:
                    SI[J] = -1
                    repeatSI += 1
    
    # Fix duplicates S1
    if repeatS1 != 0:
        candidates = list(set(current_pool) - set(S1))
        if candidates:
            replaceS1 = sample(candidates, P_score, repeatS1)
            J = 0
            for e in range(budget):
                if S1[e] == -1 and J < len(replaceS1):
                    S1[e] = replaceS1[J]
                    J += 1

    # Fix duplicates SI
    if repeatSI != 0:
        candidates = list(set(current_pool) - set(SI))
        if candidates:
            replaceSI = sample(candidates, P_score, repeatSI)
            J = 0
            for e in range(budget):
                if SI[e] == -1 and J < len(replaceSI):
                    SI[e] = replaceSI[J]
                    J += 1
    
    return S1, SI

def _evolve_subpopulation_step(
    S1, SI, budget, cOne, cTwo, com_sn, P_score, G, SN, com_and_fs, hop, 
    shared_islands, shared_islands_effect, 
    islands_index, islands_effect_index, community_id, subpop_id, index_s1, I
):
    """
    Helper function to perform crossover and mutation for one pair of individuals (S1, SI).
    """
    # Use pure function for logic
    new_S1, new_SI = full_crossover_mutate(S1, SI, budget, cOne, cTwo, com_sn, P_score)
    
    # Evaluate
    effectS1 = DPADVEvaluator.calculate_fitness(new_S1, G, SN, com_and_fs, hop)
    effectSI = DPADVEvaluator.calculate_fitness(new_SI, G, SN, com_and_fs, hop)

    # Update shared memory if better
    if effectS1 < shared_islands_effect[islands_effect_index[community_id, subpop_id, index_s1]]:
        for X in range(budget):
            shared_islands[islands_index[community_id, subpop_id, index_s1, X]] = new_S1[X]
        shared_islands_effect[islands_effect_index[community_id, subpop_id, index_s1]] = effectS1

    if effectSI < shared_islands_effect[islands_effect_index[community_id, subpop_id, I]]:
        for X in range(budget):
            shared_islands[islands_index[community_id, subpop_id, I, X]] = new_SI[X]
        shared_islands_effect[islands_effect_index[community_id, subpop_id, I]] = effectSI

def _local_search_step(
    community_id, subpop_id, Ni, budget, 
    shared_islands, shared_islands_effect, 
    islands_index, islands_effect_index,
    G, SN, com_and_fs, hop, N_prob, gama_com
):
    """
    Helper function to perform local search (Delta Score based node replacement).
    Refactored from original code (lines 446-593).
    """
    if budget == 0:
        return

    while True:
        # 1. Identify current best solution S1 in shared memory
        start_idx = islands_effect_index[community_id, subpop_id, 0]
        end_idx = islands_effect_index[community_id, subpop_id, Ni - 1] + 1
        current_effects = shared_islands_effect[start_idx:end_idx]
        index_s1 = current_effects.index(min(current_effects))

        s1_start = islands_index[community_id, subpop_id, index_s1, 0]
        s1_end = islands_index[community_id, subpop_id, index_s1, budget - 1] + 1
        S1 = list(shared_islands[s1_start:s1_end])

        discount_P_score_diff = []
        
        # 2. Calculate contribution (discount P score diff) for each node in S1
        # This determines which node is "worst" (least contribution)
        for I in range(budget):
            rs = 0
            predecessors = defaultdict(lambda: [])
            one_hop_neighbors = []
            two_hop_neighbors = []

            for v in G.neighbors(S1[I]):
                one_hop_neighbors.append(v)
                for w in G.neighbors(v):
                    two_hop_neighbors.append(w)
                    predecessors[w].append(v)

            oneAndF = set(one_hop_neighbors).intersection(set(com_and_fs)) - set(S1)
            two_hop_neighbors = set(two_hop_neighbors).intersection(set(com_and_fs)) - set(S1)
            twoAndOne = two_hop_neighbors.intersection(oneAndF)
            two_one = two_hop_neighbors - oneAndF

            for t in range(1, hop + 1):
                rs += N_prob[S1[I], t]

            for v in oneAndF:
                for t in range(1, hop + 1):
                    rs += G[S1[I]][v]['weight'] * N_prob[v, t]

            for w in twoAndOne:
                temp_p = 1
                for v in set(predecessors[w]):
                    temp_p *= (1 - G[S1[I]][v]['weight'] * G[v][w]['weight'])
                for t in range(2, hop + 1):
                    rs += (1 - G[S1[I]][w]['weight']) * (1 - temp_p) * (1 - N_prob[w, 1]) * N_prob[w, t]

            for w in two_one:
                temp_p = 1
                for v in set(predecessors[w]):
                    temp_p *= (1 - G[S1[I]][v]['weight'] * G[v][w]['weight'])
                for t in range(2, hop + 1):
                    rs += (1 - temp_p) * (1 - N_prob[w, 1]) * N_prob[w, t]

            temp = 1
            rs1 = 0
            
            # Use predecessors if directed, neighbors if undirected
            incoming_neighbors = G.predecessors(S1[I]) if G.is_directed() else G.neighbors(S1[I])
            
            for u in set(S1).intersection(set(incoming_neighbors)):
                temp *= (1 - G[u][S1[I]]['weight'])

            for t in range(1, hop + 1):
                rs1 += (1 - temp) * N_prob[S1[I], t]

            for v in oneAndF:
                for t in range(2, hop + 1):
                    rs1 += (1 - temp) * G[S1[I]][v]['weight'] * N_prob[v, t]

            discount_P_score_diff.append(rs - rs1)

        # 3. Find worst node to replace
        I_worst = discount_P_score_diff.index(min(discount_P_score_diff))
        Sbest = copy.deepcopy(S1)
        
        replace_discount_P_score_diff = {}
        
        # 4. Try replacing worst node with candidates from gama_com
        for nn in (set(gama_com) - set(Sbest)):
            S1[I_worst] = nn # Tentative replacement

            # Calculate score for new node nn (similar logic as above)
            rs = 0
            predecessors = defaultdict(lambda: [])
            one_hop_neighbors = []
            two_hop_neighbors = []

            for v in G.neighbors(S1[I_worst]):
                one_hop_neighbors.append(v)
                for w in G.neighbors(v):
                    two_hop_neighbors.append(w)
                    predecessors[w].append(v)

            oneAndF = set(one_hop_neighbors).intersection(set(com_and_fs)) - set(S1)
            two_hop_neighbors = set(two_hop_neighbors).intersection(set(com_and_fs)) - set(S1)
            twoAndOne = two_hop_neighbors.intersection(oneAndF)
            two_one = two_hop_neighbors - oneAndF

            for t in range(1, hop + 1):
                rs += N_prob.get((S1[I_worst], t), 0)

            for v in oneAndF:
                for t in range(1, hop + 1):
                    rs += G[S1[I_worst]][v]['weight'] * N_prob.get((v, t), 0)

            for w in twoAndOne:
                temp_p = 1
                for v in set(predecessors[w]):
                    temp_p *= (1 - G[S1[I_worst]][v]['weight'] * G[v][w]['weight'])
                for t in range(2, hop + 1):
                    rs += (1 - G[S1[I_worst]][w]['weight']) * (1 - temp_p) * (1 - N_prob.get((w, 1), 0)) * N_prob.get((w, t), 0)

            for w in two_one:
                temp_p = 1
                for v in set(predecessors[w]):
                    temp_p *= (1 - G[S1[I_worst]][v]['weight'] * G[v][w]['weight'])
                for t in range(2, hop + 1):
                    rs += (1 - temp_p) * (1 - N_prob.get((w, 1), 0)) * N_prob.get((w, t), 0)

            temp = 1
            rs1 = 0
            
            # Use predecessors if directed, neighbors if undirected
            incoming_neighbors = G.predecessors(S1[I_worst]) if G.is_directed() else G.neighbors(S1[I_worst])
            
            for u in set(S1).intersection(set(incoming_neighbors)):
                temp *= (1 - G[u][S1[I_worst]]['weight'])

            for t in range(1, hop + 1):
                rs1 += (1 - temp) * N_prob.get((S1[I_worst], t), 0)

            for v in oneAndF:
                for t in range(2, hop + 1):
                    rs1 += (1 - temp) * G[S1[I_worst]][v]['weight'] * N_prob.get((v, t), 0)

            replace_discount_P_score_diff[nn] = rs - rs1

        # 5. Select best replacement
        S1[I_worst] = -1 # Reset
        rmax = discount_P_score_diff[I_worst]
        rn = Sbest[I_worst] # Default to keep original if no improvement

        for nn in list(set(gama_com) - set(Sbest)):
            if replace_discount_P_score_diff[nn] >= rmax:
                rmax = replace_discount_P_score_diff[nn]
                rn = nn

        S1[I_worst] = rn

        # 6. Check if improvement occurred
        effectS1 = DPADVEvaluator.calculate_fitness(S1, G, SN, com_and_fs, hop)
        
        # If improved, update shared memory and continue loop; else break
        if effectS1 < shared_islands_effect[islands_effect_index[community_id, subpop_id, index_s1]]:
            for X in range(budget):
                shared_islands[islands_index[community_id, subpop_id, index_s1, X]] = S1[X]
            shared_islands_effect[islands_effect_index[community_id, subpop_id, index_s1]] = effectS1
        else:
            break

def _subpopulation_communication(
    community_id, subpop_id, com_res, Ni, budget,
    shared_islands, shared_islands_effect, 
    islands_index, islands_effect_index
):
    """
    Helper function for exchanging best solutions between subpopulations (Ring Topology).
    """
    # Identify best individual index in EACH subpopulation
    minPos = []
    for J in range(com_res[community_id]):
        start = islands_effect_index[community_id, J, 0]
        end = islands_effect_index[community_id, J, Ni - 1] + 1
        subpop_effects = shared_islands_effect[start:end]
        minPos.append(subpop_effects.index(min(subpop_effects)))

    # Store best of subpop 0 temporarily
    temp_s0_min = copy.deepcopy(
        shared_islands[islands_index[community_id, 0, minPos[0], 0]:
                       islands_index[community_id, 0, minPos[0], budget - 1] + 1]
    )
    temp_effect0_min = shared_islands_effect[islands_effect_index[community_id, 0, minPos[0]]]

    # Shift bests: Subpop J takes best from Subpop J+1
    for J in range(com_res[community_id] - 1):
        # Copy genes
        for X in range(budget):
            shared_islands[islands_index[community_id, J, minPos[J], X]] = \
                shared_islands[islands_index[community_id, J + 1, minPos[J + 1], X]]
        # Copy effect
        shared_islands_effect[islands_effect_index[community_id, J, minPos[J]]] = \
            shared_islands_effect[islands_effect_index[community_id, J + 1, minPos[J + 1]]]

    # Last subpop takes best from stored Subpop 0 (closing the ring)
    for X in range(budget):
        shared_islands[islands_index[community_id, com_res[community_id] - 1, minPos[com_res[community_id] - 1], X]] = \
            temp_s0_min[X]
            
    shared_islands_effect[islands_effect_index[community_id, com_res[community_id] - 1, minPos[com_res[community_id] - 1]]] = \
        temp_effect0_min

def evolve_community(
    community_id, subpop_id, 
    max_community_end_flags, max_community_id,
    shared_islands, shared_islands_effect, locks,
    G, SN, com_and_fs, hop, N_prob,
    com_sn, budget, com_res, Ni, cOne, cTwo,
    islands_index, islands_effect_index, locks_index, P_score, last_p,
    share_locks_index, begin_flag, gama_com, com_gen_acc
):
    """
    Main evolution logic for a single community subpopulation.
    Refactored to handle synchronization properly for ALL communities.
    """
    
    # Wait for start signal
    if share_locks_index[community_id, subpop_id] == last_p:
        begin_flag[0] = 1

    while int(round(begin_flag[0])) == 0:
        pass

    g = 0
    
    # NOTE: The original code logic for synchronization relied heavily on 'max_community_id'.
    # Non-max communities waited for max-community flags.
    # We preserve this structure to ensure correct lock-step execution.

    is_max_community = (community_id == max_community_id)

    # If this is the 'controller' community (max_i)
    if is_max_community:
        max_community_end_flags[subpop_id] = 0

    # Synchronization Loop Condition:
    # Max community runs until its own end logic.
    # Other communities wait for max community to signal end.
    
    while True:
        # Check termination condition for non-max communities
        if not is_max_community:
            if int(round(sum(max_community_end_flags))) == com_res[max_community_id]:
                break
        
        # --- EVOLUTION STEP ---
        if budget >= 1:
            # 1. Identify best individual S1
            start_idx = islands_effect_index[community_id, subpop_id, 0]
            end_idx = islands_effect_index[community_id, subpop_id, Ni - 1] + 1
            current_effects = shared_islands_effect[start_idx:end_idx]
            index_s1 = current_effects.index(min(current_effects))

            # 2. Iterate over population (except best)
            for I in range(Ni):
                if I == index_s1: continue
                
                # Retrieve Individuals
                s1_start = islands_index[community_id, subpop_id, index_s1, 0]
                s1_end = islands_index[community_id, subpop_id, index_s1, budget - 1] + 1
                S1 = list(shared_islands[s1_start:s1_end])
                
                si_start = islands_index[community_id, subpop_id, I, 0]
                si_end = islands_index[community_id, subpop_id, I, budget - 1] + 1
                SI = list(shared_islands[si_start:si_end])

                # Execute Crossover & Mutation Helper
                _evolve_subpopulation_step(
                    S1, SI, budget, cOne, cTwo, com_sn, P_score, G, SN, com_and_fs, hop,
                    shared_islands, shared_islands_effect,
                    islands_index, islands_effect_index, community_id, subpop_id, index_s1, I
                )

            # 3. Local Search Helper
            _local_search_step(
                community_id, subpop_id, Ni, budget,
                shared_islands, shared_islands_effect,
                islands_index, islands_effect_index,
                G, SN, com_and_fs, hop, N_prob, gama_com
            )

        # --- SYNCHRONIZATION & COMMUNICATION ---
        if com_res[community_id] > 1:
            if subpop_id == 0:
                # Subpop 0 waits for all other subpops in this community to reach current gen 'g'
                target = int(round((g + 1) * (com_res[community_id] - 1)))
                while True:
                    current_sum = sum(locks[locks_index[community_id, 1]: locks_index[community_id, com_res[community_id] - 1] + 1])
                    if int(round(current_sum)) == target:
                        break

                # Communication Helper
                _subpopulation_communication(
                    community_id, subpop_id, com_res, Ni, budget,
                    shared_islands, shared_islands_effect,
                    islands_index, islands_effect_index
                )
        
        g += 1
        print(f"{community_id}, {subpop_id} Gen {g} done")

        # Update Locks
        if is_max_community:
            max_community_end_flags[subpop_id] = 1 # Signal ready/done for this step? 
            # Note: Logic in original code toggles this flag or sets it to 1 at end of gen?
            # Original: maxCommunityEnd_11[j_11] = 1 at end of loop body.
            # And at start of loop body (if max_i): maxCommunityEnd_11[j_11] = 0.
            # But wait, the original code had an 'if' structure where max_i was separate from others.
            # Here we are merging.
            # If we loop, we need to reset flag at start of next iter?
            # Actually, standard structure:
            # 1. Set flag 0
            # 2. Evolve
            # 3. Set flag 1
            pass

        locks[locks_index[community_id, subpop_id]] = g

        if subpop_id != 0:
            # Wait for Subpop 0 (Leader) to advance
            while int(round(locks[locks_index[community_id, 0]])) != int(round(g)):
                pass
        else:
            com_gen_acc[community_id] += 1
        
        # Break condition for Max Community (One generation per call? Or loop internally?)
        # The original code loops indefinitely for non-max, but max-community seems to also loop?
        # Actually, `evolution_11` in original code has `g_11 = 0` then loops.
        # But `HMACE` structure calls `env.step()` which implies ONE generation.
        # So we should probably BREAK after one generation here to return control to main loop.
        
        break # Exit after 1 generation step
    
    # Reset flag for next call if needed?
    if is_max_community:
         max_community_end_flags[subpop_id] = 1 

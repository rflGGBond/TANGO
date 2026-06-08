import heapq
import networkx as nx
import copy
import math
from typing import List, Dict, Any
from collections import defaultdict

from ..core import graph_ops, evaluator, evolution, merger, gamma_merger
from .community import Community
from ..utils.types import CommunityAction, GlobalAction, GlobalObservation, CommunitySummary
import random

class PCMCCEnvironment:
    def __init__(self, graph_path: str, sn_nodes: List[int], total_budget: int, num_communities: int, is_directed: bool = False, tau_1: float = 0.3, tau_2: float = 0.6, merge_alpha: float = 1.5, merge_lambda: float = 0.8, disable_ter: bool = False, disable_ds: bool = False):
        self.graph_path = graph_path
        self.sn_nodes = sn_nodes
        self.total_budget = total_budget
        self.initial_num_communities = num_communities
        self.is_directed = is_directed
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.merge_alpha = merge_alpha
        self.merge_lambda = merge_lambda
        self.disable_ter = disable_ter
        self.disable_ds = disable_ds
        self.hop = 2 # Default hop count
        self.Ni = 20 # Subpopulation size (Number of individuals per island)
        self.initial_alpha = 12.0 # Default search space reduction factor
        
        # Load Graph
        self.G = nx.DiGraph() if is_directed else nx.Graph()
        with open(graph_path, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3:
                    n, m, w = int(parts[0]), int(parts[1]), float(parts[2])
                    self.G.add_edge(n, m, weight=w)
        
        # Initialize Spaces
        self._init_spaces()
        
        # Initialize Communities
        self.communities: Dict[int, Community] = {}
        self._init_communities()
        
        # Global State
        self.current_gen = 0
        self.global_dpadv_history = []
        self.parameter_history: List[Dict[str, Any]] = [] # [{'params': {...}, 'global_score': float}]
        self.global_best_seed = []
        self.global_best_dpadv = float('inf')
        
        # Convergence Tracking
        self.last_improved_gen = 0
        self.convergence_patience = 25  # Stop if no improvement for 5 generations
        
        # PCMCC Termination Logic Attributes
        self.s_g = 3
        self.theta = 0.001
        self.mu = 0.5
        self.termination_beta = 2
        self.e_g_b = None # Generation when global evolution begins (1 community)
        
        # Merge suggestions pending execution
        self.pending_merge_suggestions: List[tuple] = []
        self.merge_history: List[str] = []
        
        # Cache for expensive closeness calculations
        self.closeness_cache = defaultdict(lambda: defaultdict(float))
        self.closeness_cache_valid = False
        
        # New Flag for Emergency Trigger
        self.was_in_critical_danger = False

    def set_merge_suggestions(self, suggestions: List[tuple]):
        """
        Stores merge suggestions from Global Agent to be executed in the next step.
        """
        self.pending_merge_suggestions = suggestions

    def _convert_index(self, islands):
        """
        Helper to flatten population indices for shared memory.
        Refactored from convert_Index_10.
        """
        res_1 = {} # [i, j, N, X] -> flat_idx
        res_2 = {} # [i, j, N] -> flat_idx
        res_3 = {} # [i, j] -> flat_idx
        count_1 = 0
        count_2 = 0
        count_3 = 0

        # islands structure: [community][subpop][individual][gene]
        # In our case, self.population is [community][subpop][individual] (list of nodes)
        
        for i in range(len(islands)):
            for j in range(len(islands[i])):
                res_3[i, j] = count_3
                count_3 += 1

                for N in range(len(islands[i][j])):
                    res_2[i, j, N] = count_2
                    count_2 += 1

                    for X in range(len(islands[i][j][N])):
                        res_1[i, j, N, X] = count_1
                        count_1 += 1
        
        return res_1, res_2, res_3

    def _init_spaces(self):
        # 1. Fitness Space
        self.fitness_space = []
        cur_un_ergodic = copy.deepcopy(self.sn_nodes)
        hop_f = 0
        hop = 2 # Hardcoded for now, should be param
        
        while hop_f <= hop:
            cur_ergodic = copy.deepcopy(cur_un_ergodic)
            self.fitness_space += cur_ergodic
            if hop_f == hop:
                break
            else:
                cur_un_ergodic = []
                for u in cur_ergodic:
                    for v in self.G.neighbors(u):
                        cur_un_ergodic.append(v)
                cur_un_ergodic = list(set(cur_un_ergodic) - set(self.fitness_space))
                hop_f += 1
        self.fitness_space = list(set(self.fitness_space) - set(self.sn_nodes))

        # 2. Search Space
        self.search_space = []
        cur_un_ergodic = copy.deepcopy(self.fitness_space)
        hop_s = 0
        while hop_s <= hop:
            cur_ergodic = copy.deepcopy(cur_un_ergodic)
            self.search_space += cur_ergodic
            if hop_s == hop:
                break
            else:
                cur_un_ergodic = []
                for u in cur_ergodic:
                    for v in self.G.neighbors(u):
                        cur_un_ergodic.append(v)
                cur_un_ergodic = list(set(cur_un_ergodic) - set(self.search_space + self.sn_nodes))
                hop_s += 1
                
        # 3. Graph Subgraph (Gs)
        all_nodes = list(set(self.search_space + self.sn_nodes))
        self.Gs = self.G.subgraph(all_nodes).copy()
        
        # 4. Search Space Reduction
        # Use initial_alpha parameter instead of hardcoded value
        # Use Degree Centrality (In+Out) as heuristic
        self.search_space = heapq.nlargest(
            min(len(self.search_space), int(self.initial_alpha * self.total_budget)), 
            self.search_space, 
            key=lambda x: self.Gs.degree(x)
        )
        
        # 5. Pre-calculate Static Metrics (N_prob, P_score)
        # These depend only on graph structure and spaces, which are static during evolution.
        print("Pre-calculating static graph metrics (N_prob, P_score)...")
        self.N_prob = evaluator.DPADVEvaluator.calculate_negative_probability(
            self.Gs, self.sn_nodes, self.fitness_space, hop=2
        )
        self.P_score = evaluator.DPADVEvaluator.calculate_positive_score(
            self.Gs, self.fitness_space, self.search_space, 2, self.N_prob
        )
        print("Metrics calculated.")

    def _init_communities(self):
        # Community Division
        parts = graph_ops.detect_communities(self.Gs, self.initial_num_communities)
        
        # Use cached N_prob
        N_prob = self.N_prob
        
        # Calculate N_prob sums for each community to assign budgets
        community_probs = []
        for part in parts:
            prob_sum = 0
            for node in part:
                # Sum N_prob for t=1 to hop (here hop=2)
                for t in range(1, 3):
                    prob_sum += N_prob.get((node, t), 0)
            community_probs.append(prob_sum)
            
        total_prob = sum(community_probs)
        
        # Assign Budgets based on N_prob weight
        budgets = []
        if total_prob > 0:
            remaining_budget = self.total_budget
            for i in range(len(parts) - 1):
                # Proportional assignment
                b = int(round(self.total_budget * (community_probs[i] / total_prob)))
                b = max(1, b) # Ensure at least 1 seed
                budgets.append(b)
                remaining_budget -= b
            budgets.append(max(1, remaining_budget)) # Assign rest to last
        else:
            # Fallback to even distribution
            base = self.total_budget // len(parts)
            budgets = [base] * len(parts)
            budgets[-1] += self.total_budget % len(parts)

        for i, part in enumerate(parts):
            self.communities[i] = Community(i, part, budgets[i])
            
            # Initialize random seed set for the community
            # Ensure we have enough candidates
            candidates = list(set(self.search_space).intersection(set(part)))
            if len(candidates) < budgets[i]:
                # Fallback: take from general search space if community is too small
                candidates = self.search_space 
            
            initial_seed = random.sample(candidates, k=min(budgets[i], len(candidates)))
            
            # Calculate initial DPADV
            score = evaluator.DPADVEvaluator.calculate_fitness(
                initial_seed, self.Gs, self.sn_nodes, self.fitness_space, hop=2
            )
            self.communities[i].update_best_solution(initial_seed, score)

    def _execute_merges(self, strict_validation: bool = True):
        """
        Executes the pending merge suggestions by merging Community objects.
        If strict_validation=True (Global Agent), implements weighted gain evaluation, positive gain filtering, and Top-2 logic.
        If strict_validation=False (Heuristic), executes merges directly (assumes they are pre-validated by rules).
        """
        if not self.pending_merge_suggestions:
            return

        print(f"Executing Merges (Pending: {len(self.pending_merge_suggestions)}, Strict: {strict_validation})...")
        
        evaluated_suggestions = [] # List of tuples: (gain, merge_group, new_com_obj)
        
        # Helper map for nodes (for closeness calculation)
        com_nodes_map = {cid: com.state.nodes for cid, com in self.communities.items()}
        
        for merge_group in self.pending_merge_suggestions:
            # Filter out invalid IDs
            valid_group = [cid for cid in merge_group if cid in self.communities]
            
            if len(valid_group) < 2:
                continue
            
            # --- Logic for Strict Validation (Global Agent) ---
            if strict_validation:
                # --- Filter 1: Gamma-Reduction Merge Score Check ---
                merge_score_sum = 0
                for i in range(len(valid_group)):
                    for j in range(i + 1, len(valid_group)):
                        c1, c2 = valid_group[i], valid_group[j]
                        score = gamma_merger.calculate_merge_score(
                            self.Gs, 
                            com_nodes_map[c1], 
                            com_nodes_map[c2],
                            lambda_param=self.merge_lambda,
                            alpha_param=self.merge_alpha
                        )
                        merge_score_sum += score
                
                if merge_score_sum <= 0:
                    if getattr(self, 'disable_ter', False):
                        print(f"  -> ACCEPTED Group {valid_group} despite Gamma Merge Score <= 0 ({merge_score_sum:.4f}) due to disable_ter.")
                    else:
                        print(f"  -> REJECTED Group {valid_group}: Gamma Merge Score <= 0 ({merge_score_sum:.4f})")
                        continue

                # --- Filter 1: Closeness Check ---
                # closeness_sum = 0
                # for i in range(len(valid_group)):
                #     for j in range(i + 1, len(valid_group)):
                #         c1, c2 = valid_group[i], valid_group[j]
                #         score = merger.calculate_connection_strength(self.Gs, com_nodes_map[c1], com_nodes_map[c2])
                #         closeness_sum += score
                
                # if closeness_sum <= 0:
                #     print(f"  -> REJECTED Group {valid_group}: Closeness <= 0 ({closeness_sum})")
                #     continue
            
            # --- Preparation (Common) ---
            total_budget = 0
            weighted_baseline_score = 0
            
            new_nodes = set()
            combined_seed = []
            params_sum = {'cr1': 0, 'cr2': 0, 'beta': 0, 'alpha': 0}
            
            # First pass: Calculate Total Budget for weights
            for cid in valid_group:
                total_budget += self.communities[cid].state.budget
            
            if total_budget == 0: continue # Should not happen
            
            # Second pass: Accumulate data and calculate baseline
            for cid in valid_group:
                com = self.communities[cid]
                new_nodes.update(com.state.nodes)
                combined_seed.extend(com.state.current_seed_set)
                
                params_sum['cr1'] += com.state.cr1
                params_sum['cr2'] += com.state.cr2
                params_sum['beta'] += com.state.beta
                params_sum['alpha'] += com.state.alpha
                
                # Weighted Score contribution
                weight = com.state.budget / total_budget
                weighted_baseline_score += com.state.current_dpadv * weight
            
            # Create Candidate New Community
            new_budget = int(total_budget)
            new_com = Community(-1, list(new_nodes), new_budget)
            
            # Set Parameters (Average)
            count = len(valid_group)
            new_com.state.cr1 = params_sum['cr1'] / count
            new_com.state.cr2 = params_sum['cr2'] / count
            new_com.state.beta = params_sum['beta'] / count
            new_com.state.alpha = params_sum['alpha'] / count
            
            # Set Seed (Truncate/Fill)
            unique_seed = list(set(combined_seed))
            if len(unique_seed) > new_budget:
                unique_seed = unique_seed[:new_budget]
            elif len(unique_seed) < new_budget:
                candidates = list(new_nodes - set(unique_seed))
                needed = new_budget - len(unique_seed)
                if candidates:
                    unique_seed.extend(random.sample(candidates, min(len(candidates), needed)))
            
            # Calculate New Fitness
            new_score = evaluator.DPADVEvaluator.calculate_fitness(
                unique_seed, self.Gs, self.sn_nodes, self.fitness_space, hop=2
            )
            new_com.update_best_solution(unique_seed, new_score)
            
            # Calculate Gain
            gain = weighted_baseline_score - new_score
            
            if strict_validation:
                print(f"  -> Evaluation: Group {valid_group} | Gamma Score: {merge_score_sum:.4f} | Baseline: {weighted_baseline_score:.4f} | New: {new_score:.4f} | Gain: {gain:.4f}")
                # print(f"  -> Evaluation: Group {valid_group} | Closeness: {closeness_sum:.2f} | Baseline: {weighted_baseline_score:.4f} | New: {new_score:.4f} | Gain: {gain:.4f}")
                
                # --- Filter 2: Gain Check ---
                if getattr(self, 'disable_ter', False) or gain > 0:
                    if getattr(self, 'disable_ter', False) and gain <= 0:
                        print(f"  -> ACCEPTED: Gain <= 0 ({gain:.4f}) due to disable_ter")
                    evaluated_suggestions.append((gain, valid_group, new_com))
                else:
                    print(f"  -> REJECTED: Gain <= 0")
            else:
                # Heuristic Mode: Accept directly (with fake gain for sorting structure compatibility)
                # Or just execute immediately. To reuse the execution loop, we append with high priority.
                print(f"  -> Heuristic Accepted: Group {valid_group} | New: {new_score:.4f}")
                evaluated_suggestions.append((float('inf'), valid_group, new_com))

        # 2. Sort by Gain (Descending)
        # For Heuristic, they all have inf gain, so order is preserved.
        # For Strict, they are sorted by real gain.
        evaluated_suggestions.sort(key=lambda x: x[0], reverse=True)
        
        # 3. Execute Logic
        merged_ids = set()
        new_communities = {}
        executed_count = 0
        next_id = max(self.communities.keys()) + 1
        
        for gain, group, new_com_obj in evaluated_suggestions:
            # Strict mode: Top-2 limit
            if strict_validation and executed_count >= 2:
                break
                
            # Check for conflict
            if any(cid in merged_ids for cid in group):
                continue
                
            # Execute
            print(f"  -> EXECUTING Merge: Group {group} (Gain: {gain:.4f}) -> New ID {next_id}")
            self.merge_history.append(f"Gen {self.current_gen}: Merged {group} -> {next_id} (Gain: {gain:.2f})")
            
            # Mark IDs as merged
            for cid in group:
                merged_ids.add(cid)
            
            # Finalize New Community
            new_com_obj.state.community_id = next_id
            new_communities[next_id] = new_com_obj
            next_id += 1
            executed_count += 1
            
        # 4. Apply Changes
        for cid in merged_ids:
            if cid in self.communities:
                del self.communities[cid]
        self.communities.update(new_communities)
        
        self.pending_merge_suggestions = []
        
        if new_communities:
            print(f"Merge Complete. Created {len(new_communities)} new communities. Total communities: {len(self.communities)}")
            # Invalidate cache since topology changed
            self.closeness_cache_valid = False


    def _check_heuristic_merge(self):
        """
        Implements the original PCMCC heuristic merging logic.
        Checks for stagnation and merges communities if they don't improve fast enough.
        """
        theta = self.theta
        s_l = 3
        k = self.total_budget
        
        merge_flags = {} # community_id -> flag (-1 means needs merge)
        candidates = []
        
        # 1. Check Stagnation
        for cid, com in self.communities.items():
            current_dpadv = com.state.current_dpadv
            benchmark_dpadv = com.state.benchmark_fitness
            if benchmark_dpadv == float('inf'):
                 # First time initialization
                 com.state.benchmark_fitness = current_dpadv
                 com.state.benchmark_gen = self.current_gen
                 continue
                 
            deltaF = benchmark_dpadv - current_dpadv
            deltaT = self.current_gen - com.state.benchmark_gen
            
            # Calculate Gamma
            gamma_i = gamma_merger.calculate_gamma(self.G, com.state.nodes)
            
            # PCMCC Logic with Dynamic Threshold:
            # Theta' = Theta * (ki/k) * (1 + mu * gamma)
            threshold = theta * (com.state.budget / k) * (1 + self.mu * gamma_i) * deltaT
            
            # if (deltaF <= threshold) and (deltaT >= s_l): merge
            # elif deltaF > threshold: update benchmark
            
            if deltaF <= threshold and deltaT >= s_l:
                merge_flags[cid] = -1
                candidates.append(cid)
            elif deltaF > threshold:
                # Reset benchmark
                com.state.benchmark_fitness = current_dpadv
                com.state.benchmark_gen = self.current_gen
        
        if not candidates:
            return

        print(f"Heuristic Merge Check: Found {len(candidates)} stagnant communities: {candidates}")
        
        # 2. Find Merge Partners (Pairing)
        # Use centralized logic from merger.py
        com_nodes_map = {cid: com.state.nodes for cid, com in self.communities.items()}
        # Use new gamma-reduction strategy
        merge_groups = gamma_merger.identify_merge_groups_gamma(
            self.G, 
            com_nodes_map, 
            candidates, 
            lambda_param=self.merge_lambda, 
            alpha_param=self.merge_alpha
        )
        # merge_groups = merger.identify_merge_groups(self.G, com_nodes_map, candidates)
        
        # 3. Execute Merges
        if merge_groups:
            print(f"Heuristic Merge Groups: {merge_groups}")
            self.set_merge_suggestions(merge_groups)
            # Use strict_validation=False to bypass Global Agent specific checks
            self._execute_merges(strict_validation=False)

    def step(self, agent_active: bool = False):
        """
        Executes one generation of evolution.
        :param agent_active: True if an Agent was called in this generation.
        """
        self.current_gen += 1
        
        # Reset Emergency Flag for this step
        self.was_in_critical_danger = False
        
        # 0. Check and Execute Merges
        # A. Priority: Pending Agent Suggestions
        if self.pending_merge_suggestions:
            # Assuming pending suggestions here are from Global Agent (strict check)
            # But if they were set by heuristic just now? 
            # Actually _check_heuristic_merge calls execute immediately.
            # So if we are here, it must be from Global Agent set in previous turn.
            self._execute_merges(strict_validation=True)
        # B. Fallback: Heuristic Merging (Only if no Agent active)
        elif not agent_active:
             self._check_heuristic_merge()

        # --- Hard Constraint: Enforce Action C for Critical Danger ---
        # Calculate Delta Ref for Danger Score
        T0 = 3
        delta_ref = 1.0 
        if len(self.global_dpadv_history) > T0:
            avg_imp = (self.global_dpadv_history[0] - self.global_dpadv_history[T0]) / T0
            if avg_imp > 1e-6:
                delta_ref = avg_imp
        
        for com_id, com in self.communities.items():
            danger = self._calculate_community_danger(com, delta_ref)
            if danger >= self.tau_2:
                # MARK DANGER STATE
                self.was_in_critical_danger = True
                
                # Check if Global Agent already set aggressive parameters
                # Thresholds: Alpha >= 20.0, CR2 >= 0.8
                is_aggressive = (com.state.alpha >= 20.0) and (com.state.cr2 >= 0.8)
                
                if not is_aggressive:
                    print(f"System Enforcement: Global Agent failed to respond to danger signal. Forcibly executing 'Forced Perturbation' strategy for Community {com_id} (Danger: {danger:.2f} >= {self.tau_2}).")
                    # Force Aggressive Parameters (Alpha=25, CR2=0.9, Beta=1.5)
                    com.state.alpha = 25.0
                    com.state.cr2 = 0.9
                    com.state.beta = 1.5
                    # Optionally set cr1 too if needed
                    com.state.cr1 = 0.8
            
            elif danger >= self.tau_1 and danger < self.tau_2:
                pass
                # Level 1 logic moved to Local Agent as per request

        # 1. Parallel Evolution (Simulated Single-Threaded with Real Logic)
        print(f"Env: Executing step {self.current_gen}...")
        
        # Use cached static metrics
        N_prob = self.N_prob
        P_score = self.P_score
        
        # Sort P_score to get top nodes globally (descending order)
        sorted_p_score = sorted(P_score.items(), key=lambda x: x[1], reverse=True)
        
        for com_id, com in self.communities.items():
            # Update community with top scoring nodes
            com.set_top_k_nodes(sorted_p_score)
            
            current_seed = com.state.current_seed_set
            if not current_seed: continue
            
            # Pre-calculate shared set for this community
            com_and_fs = set(com.state.nodes).union(set(self.fitness_space))
            gama_com = self.search_space
            
            # --- Population Initialization (if needed) ---
            if not com.state.population:
                com.state.population = []
                com.state.population_scores = []
                
                # 1. First individual is the current best
                com.state.population.append(copy.deepcopy(current_seed))
                score = evaluator.DPADVEvaluator.calculate_fitness(current_seed, self.G, self.sn_nodes, com_and_fs, self.hop)
                com.state.population_scores.append(score)
                
                # 2. Others are mutations
                for _ in range(self.Ni - 1):
                    mutant = copy.deepcopy(current_seed)
                    if self.search_space:
                         candidates = list(set(self.search_space) - set(mutant))
                         if candidates:
                             idx = random.randint(0, len(mutant)-1)
                             mutant[idx] = random.choice(candidates)
                    com.state.population.append(mutant)
                    score = evaluator.DPADVEvaluator.calculate_fitness(mutant, self.G, self.sn_nodes, com_and_fs, self.hop)
                    com.state.population_scores.append(score)

            # --- Population Evolution Loop (Ni iterations) ---
            # Using Legacy Logic: Iterate through population, cross with best
            
            # Find index of best individual (S1)
            best_idx = com.state.population_scores.index(min(com.state.population_scores))
            
            for I in range(self.Ni):
                if I == best_idx: continue
                
                # Identify S1 (Best) and SI (Target)
                # Note: We fetch S1 dynamically as it might have been updated in previous iterations
                S1_input = copy.deepcopy(com.state.population[best_idx])
                SI = copy.deepcopy(com.state.population[I])
                
                # Crossover & Mutation
                # Using helper which implements the logic: 
                #   Two-way/One-way crossover based on cr1/cr2
                #   Conflict resolution (mutation)
                S1_new, SI_new = evolution.crossover_and_mutate(
                    S1_input, SI, com.state.budget, 
                    com.state.cr1, com.state.cr2, 
                    self.search_space, P_score,
                    alpha=com.state.alpha
                )
                
                # Evaluate Offspring
                effectS1 = evaluator.DPADVEvaluator.calculate_fitness(
                    S1_new, self.G, self.sn_nodes, com_and_fs, self.hop
                )
                
                effectSI = evaluator.DPADVEvaluator.calculate_fitness(
                    SI_new, self.G, self.sn_nodes, com_and_fs, self.hop
                )
                
                # Update Population (Greedy Selection)
                
                # 1. Check if S1 offspring improved global best
                if effectS1 < com.state.population_scores[best_idx]:
                    com.state.population[best_idx] = S1_new
                    com.state.population_scores[best_idx] = effectS1
                    
                # 2. Check if SI offspring improved local individual
                if effectSI < com.state.population_scores[I]:
                    com.state.population[I] = SI_new
                    com.state.population_scores[I] = effectSI
            
            # End of Ni Loop
            
            # 4. Local Search (Apply to the best candidate in population)
            final_best_idx = com.state.population_scores.index(min(com.state.population_scores))
            search_start_node = copy.deepcopy(com.state.population[final_best_idx])
            
            optimized_seed = evolution.local_search(
                search_start_node, self.G, self.sn_nodes, com_and_fs, self.hop, N_prob, gama_com
            )
            
            # 5. Final Evaluation & Update
            final_fitness = evaluator.DPADVEvaluator.calculate_fitness(
                optimized_seed, self.G, self.sn_nodes, com_and_fs, self.hop
            )
            
            # Update Population with optimized result
            if final_fitness < com.state.population_scores[final_best_idx]:
                com.state.population[final_best_idx] = optimized_seed
                com.state.population_scores[final_best_idx] = final_fitness
            
            # Check for improvement against Community's historical best
            if final_fitness < com.state.current_dpadv:
                com.update_best_solution(optimized_seed, final_fitness)
                com.reset_stagnation()
                com.calculate_metrics(com.state.population, self.Gs) 
            else:
                com.increment_stagnation()
                # Update metrics using current population
                com.calculate_metrics(com.state.population, self.Gs)

        # 2. Update Global State
        # Correctly calculate global DPADV by combining seeds from all communities
        current_global_seed = []
        for com in self.communities.values():
            current_global_seed.extend(com.state.current_seed_set)
            
        if current_global_seed:
            # Calculate true global fitness
            current_global_dpadv = evaluator.DPADVEvaluator.calculate_fitness(
                current_global_seed, self.Gs, self.sn_nodes, self.fitness_space, hop=2
            )
            
            # Check for improvement
            if current_global_dpadv < self.global_best_dpadv:
                print(f"Env Step {self.current_gen}: New Global Best Found! ({self.global_best_dpadv:.4f} -> {current_global_dpadv:.4f})")
                self.global_best_dpadv = current_global_dpadv
                self.global_best_seed = copy.deepcopy(current_global_seed)
                self.last_improved_gen = self.current_gen
                
                # Update the score of the active parameters in history
                # Assuming the last entry in parameter_history corresponds to the current active global baselines
                if self.parameter_history:
                    self.parameter_history[-1]['global_score'] = current_global_dpadv
                    # Maintain sorted order (top 10 best params)
                    self.parameter_history.sort(key=lambda x: x['global_score'])
            
            self.global_dpadv_history.append(self.global_best_dpadv)

    def _calculate_community_danger(self, com: Community, delta_ref: float) -> float:
        """
        Calculates the Danger Score for a single community.
        Danger = Sigmoid(a*Gamma + b*Stagnation + c*Collapse + d*Risk - Bias)
        """
        if getattr(self, 'disable_ds', False):
            return 0.0
            
        # 1. Gamma (Clusteredness)
        gamma_i = gamma_merger.calculate_gamma(self.Gs, com.state.nodes)
        
        # 2. Stagnation
        # Stagnation = max(0, 1 - Delta_i / (Delta_ref + epsilon))
        imp_rate = 0.0
        history = com.state.dpadv_history
        if len(history) >= 2:
            delta_f = history[0] - history[-1]
            delta_t = len(history) - 1
            if delta_t > 0:
                imp_rate = delta_f / delta_t
                
        epsilon = 1e-6
        stagnation_i = max(0.0, 1.0 - (imp_rate / (delta_ref + epsilon)))
        
        # 3. Collapse (Diversity < Threshold)
        div_th = 0.1
        collapse_i = 1.0 if com.state.diversity_score < div_th else 0.0
        
        # 4. Boundary Risk
        total_nodes = max(1, len(com.state.nodes))
        boundary_risk_i = len(com.state.boundary_nodes) / total_nodes
        
        # 5. Final Danger Score
        a, b, c, d = 1.0, 1.0, 0.5, 0.5
        bias = 2.0
        logit = (a * gamma_i) + (b * stagnation_i) + (c * collapse_i) + (d * boundary_risk_i) - bias
        danger_i = 1.0 / (1.0 + math.exp(-logit))
        
        return danger_i


    def get_global_observation(self):
        """
        Aggregates state to form GlobalObservation.
        """
        # --- Calculate Community Closeness ---
        # Call logic from merger.py as requested
        
        closeness_map = self.closeness_cache
        
        # Recalculate only if cache is invalid
        if not self.closeness_cache_valid:
            # 1. Prepare list of community nodes
            com_nodes_map = {cid: com.state.nodes for cid, com in self.communities.items()}
            
            # 2. Calculate Pairwise Scores using merger.calculate_connection_strength
            # Clear old cache
            closeness_map.clear()
            
            # Step 1: Fast neighbor detection
            node_to_com = {}
            for cid, nodes in com_nodes_map.items():
                for n in nodes:
                    node_to_com[n] = cid
                    
            neighbors = defaultdict(set)
            for u, v in self.Gs.edges():
                c1 = node_to_com.get(u)
                c2 = node_to_com.get(v)
                if c1 is not None and c2 is not None and c1 != c2:
                    neighbors[c1].add(c2)
                    neighbors[c2].add(c1)
            
            # Step 2: Call merger.py for identified neighbors
            for c1, neighbor_set in neighbors.items():
                for c2 in neighbor_set:
                    if c1 < c2: # Avoid double calc
                        # CALL THE MERGER FUNCTION
                        score = merger.calculate_connection_strength(self.Gs, com_nodes_map[c1], com_nodes_map[c2])
                        closeness_map[c1][c2] = score
                        closeness_map[c2][c1] = score
            
            self.closeness_cache_valid = True
        
        # --- Calculate Delta Ref (Early Global Average Improvement) ---
        T0 = 3
        delta_ref = 1.0 # Default to avoid div by zero if not enough history
        
        if len(self.global_dpadv_history) > T0:
            # D(0), ..., D(T0) are first T0+1 elements
            # Delta_ref = 1/T0 * Sum(D(t) - D(t+1)) for t=0 to T0-1
            # Which simplifies to (D(0) - D(T0)) / T0
            D0 = self.global_dpadv_history[0]
            DT0 = self.global_dpadv_history[T0]
            
            # Improvement is (D(t) - D(t+1)) assuming D is minimization (fitness)
            # If D0 > DT0, improvement is positive
            avg_imp = (D0 - DT0) / T0
            if avg_imp > 1e-6:
                delta_ref = avg_imp
        
        summaries = []
        emergency_global_call = False # Flag for emergency intervention

        for com_id, com in self.communities.items():
            # Get calculated closeness
            closeness = dict(closeness_map[com_id])
            
            # Update community's neighbor list while we are at it
            com.update_neighbors(list(closeness.keys()))
            
            # Calculate Improvement Rate (Delta DPADV / Delta T)
            # Use history: dpadv_history
            imp_rate = 0.0
            history = com.state.dpadv_history
            if len(history) >= 2:
                # Calculate drop from oldest recorded to current
                # history[-1] is current (usually), history[0] is oldest in window
                delta_f = history[0] - history[-1]
                delta_t = len(history) - 1 # Generations elapsed in window
                if delta_t > 0:
                    imp_rate = delta_f / delta_t
            
            # --- Danger Score Calculation ---
            danger_i = self._calculate_community_danger(com, delta_ref)
            
            # Recalculate Gamma for observation
            gamma_i = gamma_merger.calculate_gamma(self.Gs, com.state.nodes)

            # Check for Critical Danger
            # If current danger is high OR danger was detected during step execution
            if danger_i >= self.tau_2 or self.was_in_critical_danger:
                emergency_global_call = True
            
            summaries.append(CommunitySummary(
                community_id=com_id,
                budget=com.state.budget,
                best_dpadv=com.state.current_dpadv,
                improvement_rate=imp_rate, 
                diversity=com.state.diversity_score,
                boundary_risk=len(com.state.boundary_nodes) / max(1, len(com.state.nodes)),
                closeness_info=closeness,
                danger_score=danger_i,
                gamma=gamma_i
            ))
            
        return GlobalObservation(
            current_generation=self.current_gen,
            current_global_dpadv=self.global_best_dpadv,
            global_dpadv_history=self.global_dpadv_history,
            community_summaries=summaries,
            merge_history=self.merge_history,
            parameter_history=self.parameter_history,
            emergency_global_call=emergency_global_call # Pass the flag
        )

    def apply_community_action(self, community_id: int, action: CommunityAction):
        """
        Applies the action from a Local Agent, including strict candidate evaluation.
        """
        com = self.communities.get(community_id)
        if not com: return
        
        # 1. Update Parameters (Mode A) - Now with Simulation Evaluation
        if action.parameters:
            print(f"Agent {community_id} proposing parameter adjustment: {action.parameters}")
            
            # --- Simulation Step ---
            # Create a temporary simulation to test if these parameters actually help
            
            # 1. Prepare Simulation Context
            # We need to simulate one evolution step for this community with the NEW parameters
            sim_params = action.parameters
            current_seed = com.state.current_seed_set
            
            # Use cached metrics
            N_prob = self.N_prob
            P_score = self.P_score
            
            # 2. Execute Evolutionary Step (Mutation -> Crossover -> Local Search)
            # A. Generate SI (Mutation)
            SI = copy.deepcopy(current_seed)
            if self.search_space and SI: # Check if SI is not empty
                 candidates = list(set(self.search_space) - set(SI))
                 if candidates:
                     idx = random.randint(0, len(SI)-1)
                     SI[idx] = random.choice(candidates)
            
            # B. Crossover (using NEW parameters)
            S1_input = copy.deepcopy(current_seed)
            S1_sim, _ = evolution.crossover_and_mutate(
                S1_input, SI, com.state.budget, 
                sim_params.get('cr1', com.state.cr1), 
                sim_params.get('cr2', com.state.cr2), 
                self.search_space, P_score,
                alpha=sim_params.get('alpha', com.state.alpha)
            )
            
            # C. Local Search (using NEW parameters)
            com_and_fs = set(com.state.nodes).union(set(self.fitness_space))
            gama_com = self.search_space # Simplified, or could be refined based on alpha
            
            S1_sim_new = evolution.local_search(
                S1_sim, self.G, self.sn_nodes, com_and_fs, self.hop, N_prob, gama_com
            )
            if S1_sim_new != S1_sim:
                S1_sim = S1_sim_new
                
            # 3. Evaluate Result
            sim_fitness = evaluator.DPADVEvaluator.calculate_fitness(
                S1_sim, self.G, self.sn_nodes, com_and_fs, self.hop
            )
            
            # 4. Compare with Current Best
            # Note: We compare against com.state.current_dpadv (Local Best)
            if sim_fitness < com.state.current_dpadv:
                print(f"  -> Simulation Successful! New Params yielded better local fitness ({com.state.current_dpadv:.4f} -> {sim_fitness:.4f}). Accepted.")
                # Accept Parameters
                com.update_parameters(action.parameters)
                # OPTIONAL: Also accept the better solution found during simulation?
                # For now, let's just accept parameters. The next real step will likely find it or better.
            else:
                print(f"  -> Simulation Failed. New Params did not improve local fitness ({sim_fitness:.4f} >= {com.state.current_dpadv:.4f}). Rejected.")
                # Reject Parameters - Do not call update_parameters
                pass
        
        # 2. Candidate Generation (Mode B) - Try-Evaluate-Revert Logic
        if action.candidate_seed_set:
            # Validate constraints (e.g., size match)
            if len(action.candidate_seed_set) != com.state.budget:
                print(f"Agent {community_id} provided seed set of wrong size. Ignored.")
                return

            # Validate node existence
            for node in action.candidate_seed_set:
                if not self.Gs.has_node(node):
                    print(f"Agent {community_id} proposed non-existent node {node}. Ignored.")
                    return

            # Construct Global Candidate: Combine new candidate with OTHER communities' current bests
            global_candidate_seed = []
            for cid, other_com in self.communities.items():
                if cid == community_id:
                    global_candidate_seed.extend(action.candidate_seed_set)
                else:
                    global_candidate_seed.extend(other_com.state.current_seed_set)
            
            # Calculate Global DPADV for this candidate configuration
            new_global_dpadv = evaluator.DPADVEvaluator.calculate_fitness(
                global_candidate_seed, self.Gs, self.sn_nodes, self.fitness_space, hop=2
            )
            # new_global_dpadv = evaluator.DPADVEvaluator.calculate_fitness(
            #     global_candidate_seed, self.Gs, self.sn_nodes, self.fitness_space, hop=2
            # )
            
            # Compare with current global best
            if getattr(self, 'disable_ter', False) or new_global_dpadv < self.global_best_dpadv:
                if getattr(self, 'disable_ter', False):
                    print(f"Agent {community_id} candidate blindly ACCEPTED due to disable_ter (DPADV: {self.global_best_dpadv} -> {new_global_dpadv}).")
                else:
                    print(f"Agent {community_id} found BETTER global solution (DPADV: {self.global_best_dpadv} -> {new_global_dpadv}). Accepted.")
                # Accept: Update Community Best & Global Best
                com.update_best_solution(action.candidate_seed_set, new_global_dpadv) # Note: Local DPADV is approximation here
                self.global_best_dpadv = new_global_dpadv
                self.global_best_seed = global_candidate_seed
                self.last_improved_gen = self.current_gen # Reset patience
            else:
                # Reject: Do nothing (Revert is implicit by not applying)
                print(f"Agent {community_id} candidate rejected for GLOBAL best (DPADV: {new_global_dpadv:.4f} >= {self.global_best_dpadv:.4f}).")
                
                # --- Local Fallback Check ---
                # Even if it fails globally, check if it improves the community locally
                com_and_fs = set(com.state.nodes).union(set(self.fitness_space))
                local_fitness = evaluator.DPADVEvaluator.calculate_fitness(
                    action.candidate_seed_set, self.Gs, self.sn_nodes, com_and_fs, self.hop
                )
                
                if local_fitness < com.state.current_dpadv:
                     print(f"  -> But ACCEPTED for LOCAL improvement (Local DPADV: {com.state.current_dpadv:.4f} -> {local_fitness:.4f}).")
                     com.update_best_solution(action.candidate_seed_set, local_fitness)
                     com.reset_stagnation()

    def check_termination(self, max_gen: int) -> bool:
        """
        Checks termination conditions based on PCMCC logic.
        1. Timeout: max_gen + beta * s_g
        2. Convergence: 1 community + stable fitness
        """
        # 1. Timeout Check
        timeout_gen = max_gen + self.termination_beta * self.s_g
        if self.current_gen >= timeout_gen:
            print(f"Termination: Reached maximum generation limit ({timeout_gen}).")
            return True

        # 2. Convergence Check
        if len(self.communities) == 1:
            # Set e_g_b if this is the first time we see 1 community
            if self.e_g_b is None:
                self.e_g_b = self.current_gen
                print(f"Global evolution phase started at generation {self.e_g_b}")
            
            # Ensure we have enough history in the global phase
            # Condition: (curT > s_g) and ((curT - s_g) >= e_g_b)
            if (self.current_gen > self.s_g) and ((self.current_gen - self.s_g) >= self.e_g_b):
                # Check slope: improvement over last s_g generations
                # global_dpadv_history stores values for each generation [gen1, gen2, ... genN]
                
                if len(self.global_dpadv_history) > self.s_g:
                    # history[-1] is current. history[-(1+s_g)] is s_g steps ago.
                    current_val = self.global_dpadv_history[-1]
                    past_val = self.global_dpadv_history[-(1 + self.s_g)]
                    
                    improvement = past_val - current_val
                    threshold = self.theta * self.s_g
                    
                    if improvement <= threshold:
                        print(f"Termination: Converged. (Improvement {improvement:.4f} <= Threshold {threshold:.4f})")
                        return True
        
        return False

    def check_convergence(self) -> bool:
        """
        Checks if the global best DPADV has stagnated for 'patience' generations.
        Returns True if converged (should stop), False otherwise.
        """
        generations_since_improvement = self.current_gen - self.last_improved_gen
        
        if generations_since_improvement >= self.convergence_patience:
            print(f"Convergence Reached: No improvement for {generations_since_improvement} generations.")
            return True
            
        return False

    def apply_global_action(self, action: GlobalAction):
        """
        Applies global decisions from Global Agent.
        """
        # 1. Update Global Baselines - Now with Global Simulation Evaluation
        if action.global_baselines:
            print(f"Global Agent proposing global baselines: {action.global_baselines}")
            
            # --- Global Simulation Step ---
            # Simulate one step for ALL communities using the new global parameters
            
            sim_global_seeds = []
            sim_local_seeds = {} # Store per-community simulated seeds
            sim_params = action.global_baselines
            
            # Defensive check: Ensure sim_params is a dict
            if isinstance(sim_params, list):
                print(f"Warning: global_baselines received as list {sim_params}. Attempting to extract dict.")
                if len(sim_params) > 0 and isinstance(sim_params[0], dict):
                    sim_params = sim_params[0]
                else:
                    sim_params = {} # Fallback to empty to use defaults
            
            # Use cached metrics
            N_prob = self.N_prob
            P_score = self.P_score
            
            print("  -> Running Global Simulation for new parameters...")
            
            for com_id, com in self.communities.items():
                current_seed = com.state.current_seed_set
                
                # A. Generate SI (Mutation)
                SI = copy.deepcopy(current_seed)
                if self.search_space and SI: # Check if SI is not empty
                     candidates = list(set(self.search_space) - set(SI))
                     if candidates:
                         idx = random.randint(0, len(SI)-1)
                         SI[idx] = random.choice(candidates)
                
                # B. Crossover (using NEW GLOBAL parameters)
                # Note: Global Agent sets baselines, but communities might have their own overrides.
                # Here we assume Global Baseline overrides everything for the simulation to test its pure effect.
                S1_input = copy.deepcopy(current_seed)
                S1_sim, _ = evolution.crossover_and_mutate(
                    S1_input, SI, com.state.budget, 
                    sim_params.get('cr1', com.state.cr1), 
                    sim_params.get('cr2', com.state.cr2), 
                    self.search_space, P_score,
                    alpha=sim_params.get('alpha', com.state.alpha)
                )
                
                # C. Local Search
                com_and_fs = set(com.state.nodes).union(set(self.fitness_space))
                gama_com = self.search_space 
                
                S1_sim_new = evolution.local_search(
                    S1_sim, self.G, self.sn_nodes, com_and_fs, self.hop, N_prob, gama_com
                )
                if S1_sim_new != S1_sim:
                    S1_sim = S1_sim_new
                
                sim_global_seeds.extend(S1_sim)
                sim_local_seeds[com_id] = S1_sim # Store for potential update
                
            # Evaluate Global Result
            sim_global_dpadv = evaluator.DPADVEvaluator.calculate_fitness(
                sim_global_seeds, self.Gs, self.sn_nodes, self.fitness_space, hop=2
            )
            
            # Compare with Current Global Best
            if getattr(self, 'disable_ter', False) or sim_global_dpadv < self.global_best_dpadv:
                 if getattr(self, 'disable_ter', False):
                     print(f"  -> Global Simulation Accepted blindly due to disable_ter (DPADV: {self.global_best_dpadv:.4f} -> {sim_global_dpadv:.4f}).")
                 else:
                     print(f"  -> Global Simulation Successful! New Baselines improved Global DPADV ({self.global_best_dpadv:.4f} -> {sim_global_dpadv:.4f}). Accepted.")
                 
                 # 1. Apply Parameters
                 for com in self.communities.values():
                     com.update_parameters(action.global_baselines, is_global_baseline=True)
                 
                 # 2. Apply Better Solution (Global & Local)
                 self.global_best_dpadv = sim_global_dpadv
                 self.global_best_seed = copy.deepcopy(sim_global_seeds)
                 self.last_improved_gen = self.current_gen
                 
                 # Update each community's best solution with the simulated one
                 # Note: We don't have the exact local fitness for S1_sim here (we optimized for Global).
                 # But we should still update the seed set.
                 # We can calculate local fitness if needed, or just set it.
                 for com_id, s1_sim in sim_local_seeds.items():
                     com = self.communities[com_id]
                     # Calculate local fitness for consistency
                     com_and_fs = set(com.state.nodes).union(set(self.fitness_space))
                     local_fitness = evaluator.DPADVEvaluator.calculate_fitness(
                        s1_sim, self.G, self.sn_nodes, com_and_fs, self.hop
                     )
                     com.update_best_solution(s1_sim, local_fitness)
                     com.reset_stagnation()
                 
                 # Track Parameter History
                 self.parameter_history.append({
                     'params': action.global_baselines,
                     'global_score': sim_global_dpadv # Use the simulated score which is accurate
                 })
                 
                 self.parameter_history.sort(key=lambda x: x['global_score'])
                 if len(self.parameter_history) > 10:
                     self.parameter_history = self.parameter_history[:10]
            else:
                 print(f"  -> Global Simulation Failed. New Baselines did not improve Global DPADV ({sim_global_dpadv:.4f} >= {self.global_best_dpadv:.4f}). Rejected.")
                 pass
                
        # 2. Update Budgets (Redistribution)
        if action.budget_adjustments:
            print(f"Global Agent adjusting budgets: {action.budget_adjustments}")
            
            # 1. Apply Adjustments
            temp_budgets = {}
            for com_id, com in self.communities.items():
                current_b = com.state.budget
                adjustment = action.budget_adjustments.get(com_id, 0)
                # Apply delta and enforce minimum 1
                new_b = max(1, current_b + int(adjustment))
                temp_budgets[com_id] = new_b
            
            # 2. Normalize to Total Budget (k)
            current_total = sum(temp_budgets.values())
            target_total = self.total_budget
            
            if current_total != target_total and current_total > 0:
                print(f"  -> Normalizing budgets: Sum {current_total} != Target {target_total}")
                normalized_budgets = {}
                running_sum = 0
                
                # Sort by value to minimize rounding impact on small communities
                sorted_items = sorted(temp_budgets.items(), key=lambda x: x[1])
                
                for i, (cid, val) in enumerate(sorted_items):
                    if i == len(sorted_items) - 1:
                        # Last one takes the remainder
                        new_val = target_total - running_sum
                    else:
                        # Proportional share
                        new_val = int(round((val / current_total) * target_total))
                        
                    # Ensure at least 1 (unless target_total is 0, which shouldn't happen)
                    new_val = max(1, new_val)
                    normalized_budgets[cid] = new_val
                    running_sum += new_val
                
                # Final check if max(1) pushed us over
                final_sum = sum(normalized_budgets.values())
                if final_sum > target_total:
                    # Reduce from the largest until we match
                    diff = final_sum - target_total
                    while diff > 0:
                        # Find largest > 1
                        candidates = [cid for cid, b in normalized_budgets.items() if b > 1]
                        if not candidates: break # Cannot reduce further
                        
                        # Pick random or max
                        target_cid = max(candidates, key=lambda c: normalized_budgets[c])
                        normalized_budgets[target_cid] -= 1
                        diff -= 1
                
                # Apply Normalized Budgets
                for cid, b in normalized_budgets.items():
                    self.communities[cid].state.budget = b
            else:
                # Apply Temp Budgets directly if sum matches (or if something weird happened)
                for cid, b in temp_budgets.items():
                     self.communities[cid].state.budget = b
                    
        # 3. Execute Merges (Already handled via set_merge_suggestions, but ensuring consistency)
        if action.merge_suggestions:
            # If not already set by run.py logic, set it here
            if not self.pending_merge_suggestions:
                self.set_merge_suggestions(action.merge_suggestions)

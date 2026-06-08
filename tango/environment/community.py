from dataclasses import dataclass, field
from typing import List, Dict, Set, Any, Optional
import copy
import numpy as np

@dataclass
class CommunityState:
    community_id: int
    nodes: List[int]
    budget: int
    
    # Evolution Parameters (Base + Adjustment)
    base_cr1: float = 0.3
    base_cr2: float = 0.3
    base_beta: float = 2.0
    base_alpha: float = 12.0
    
    cr1: float = 0.3
    cr2: float = 0.3
    beta: float = 2.0  # Local search intensity / maxT factor
    alpha: float = 12.0 # Search space reduction factor
    
    # Current Solution
    current_seed_set: List[int] = field(default_factory=list)
    current_dpadv: float = float('inf')
    
    # Population (for diversity)
    population: List[List[int]] = field(default_factory=list)
    population_scores: List[float] = field(default_factory=list)
    
    # History
    dpadv_history: List[float] = field(default_factory=list)
    solution_history: List[Dict[str, Any]] = field(default_factory=list) # List of {'seed': [], 'score': float}
    
    # Metrics
    diversity_score: float = 0.0
    top_k_score_nodes: List[Dict[str, Any]] = field(default_factory=list) # [{'id': 1, 'score': 0.5}, ...]
    
    # Boundary Info (Simplified for now)
    neighbor_community_ids: List[int] = field(default_factory=list)
    boundary_nodes: List[int] = field(default_factory=list)
    
    # Status
    stage: str = "exploration" # exploration, exploitation, stagnation
    stagnation_count: int = 0 # Generations without improvement
    
    # Heuristic Baseline (for automatic merging)
    benchmark_gen: int = 0
    benchmark_fitness: float = float('inf')

class Community:
    def __init__(self, community_id: int, nodes: List[int], budget: int):
        self.state = CommunityState(
            community_id=community_id,
            nodes=nodes,
            budget=budget,
            benchmark_fitness=float('inf') # Initialize high
        )
        # Population is managed by the environment's executor usually, 
        # but we can keep a reference or local copy if needed.
        # For now, we assume the Environment passes the population to the evaluator,
        # and updates the Community object with the best result.
    
    def reset_stagnation(self):
        self.state.stagnation_count = 0
        
    def increment_stagnation(self):
        self.state.stagnation_count += 1
    
    def update_best_solution(self, seed_set: List[int], dpadv: float):
        self.state.current_seed_set = copy.deepcopy(seed_set)
        self.state.current_dpadv = dpadv
        self.state.dpadv_history.append(dpadv)
        
        # Add to solution history
        new_solution = {'seed': copy.deepcopy(seed_set), 'score': dpadv}
        # Avoid duplicates
        if not any(s['seed'] == seed_set for s in self.state.solution_history):
             self.state.solution_history.append(new_solution)
        
        # Sort by score (ascending for DPADV - lower is better)
        self.state.solution_history.sort(key=lambda x: x['score'])
        
        # Keep top 10 solutions
        if len(self.state.solution_history) > 10:
             self.state.solution_history = self.state.solution_history[:10]
        
        # Maintain history length (e.g., last 20 gens)
        if len(self.state.dpadv_history) > 5:
            self.state.dpadv_history.pop(0)

    def update_parameters(self, params: Dict[str, Any], is_global_baseline: bool = False):
        """
        Updates parameters. If is_global_baseline is True, updates the base values.
        If False (Agent action), updates the active values (overriding base).
        """
        if is_global_baseline:
            if 'cr1' in params: self.state.base_cr1 = params['cr1']
            if 'cr2' in params: self.state.base_cr2 = params['cr2']
            if 'beta' in params: self.state.base_beta = params['beta']
            if 'alpha' in params: self.state.base_alpha = params['alpha']
            
            # Reset active parameters to new base (unless we want to preserve local drift, 
            # but usually global baseline change implies a reset)
            self.state.cr1 = self.state.base_cr1
            self.state.cr2 = self.state.base_cr2
            self.state.beta = self.state.base_beta
            self.state.alpha = self.state.base_alpha
            
        else:
            # Local adjustment
            if 'cr1' in params: self.state.cr1 = params['cr1']
            if 'cr2' in params: self.state.cr2 = params['cr2']
            if 'beta' in params: self.state.beta = params['beta']
            if 'alpha' in params: self.state.alpha = params['alpha']
        
    def set_top_k_nodes(self, nodes_with_scores: List[tuple]):
        # nodes_with_scores: [(node_id, score), ...]
        self.state.top_k_score_nodes = [
            {'id': n, 'score': s} for n, s in nodes_with_scores
        ]

    def calculate_metrics(self, population: List[List[int]], G):
        """
        Calculates diversity and boundary info based on the current population and graph.
        """
        # 1. Diversity: Average Jaccard Distance between individuals in population
        if not population or len(population) < 2:
            self.state.diversity_score = 0.0
        else:
            total_dist = 0
            count = 0
            # Sample a few pairs to estimate diversity if population is large
            num_samples = min(50, len(population))
            sampled_indices = np.random.choice(len(population), num_samples, replace=False)
            
            for i in range(len(sampled_indices)):
                for j in range(i + 1, len(sampled_indices)):
                    set_a = set(population[sampled_indices[i]])
                    set_b = set(population[sampled_indices[j]])
                    intersection = len(set_a.intersection(set_b))
                    union = len(set_a.union(set_b))
                    if union > 0:
                        total_dist += (1 - intersection / union)
                    count += 1
            
            if count > 0:
                self.state.diversity_score = total_dist / count
            else:
                self.state.diversity_score = 0.0

        # 2. Boundary Info
        # Find neighbors of community nodes that are NOT in this community
        community_nodes_set = set(self.state.nodes)
        boundary_nodes = set()
        
        # This calculation can be expensive, so maybe do it only periodically or simplified
        # For now, we assume G is available.
        # Ideally, 'neighbor_community_ids' should be passed from Environment because Community doesn't know others.
        # So we only calculate 'boundary_nodes' (nodes in this community that have edges to outside)
        
        is_directed = G.is_directed() if hasattr(G, "is_directed") else False

        for u in self.state.nodes:
            is_boundary = False
            
            # Use chain or just iterate twice to avoid creating large lists
            # 1. Outgoing / Undirected Neighbors
            if not is_boundary:
                for v in G.neighbors(u):
                    if v not in community_nodes_set:
                        is_boundary = True
                        break
            
            # 2. Incoming Neighbors (Only for Directed)
            if is_directed and not is_boundary:
                for v in G.predecessors(u):
                    if v not in community_nodes_set:
                        is_boundary = True
                        break
                        
            if is_boundary:
                boundary_nodes.add(u)
        
        self.state.boundary_nodes = list(boundary_nodes)
        
    def update_neighbors(self, neighbor_ids: List[int]):
        self.state.neighbor_community_ids = neighbor_ids

    def get_observation(self, current_gen: int, global_stage: str, global_dpadv: float) -> Dict[str, Any]:
        """
        Generates the observation dictionary for the Local Agent (LLM).
        """
        import math
        
        # Dynamic Top-K Calculation
        # Formula: Ki = clip([3ki], 10, [0.15ni])
        n_i = len(self.state.nodes)
        k_i = self.state.budget
        
        target = math.ceil(3 * k_i)
        lower_bound = 10
        upper_bound = math.ceil(0.15 * n_i)
        
        # Priority on upper bound (if community is small, don't exceed 15%)
        dynamic_k = int(min(upper_bound, max(lower_bound, target)))
        
        # Safety for extremely small communities (e.g., < 7 nodes where 15% < 1)
        if dynamic_k < 1 and n_i > 0:
            dynamic_k = 1
            
        return {
            "community_id": self.state.community_id,
            "current_generation": current_gen,
            "global_stage": global_stage,
            "budget": self.state.budget,
            "current_dpadv": self.state.current_dpadv,
            "dpadv_history": self.state.dpadv_history,
            "solution_history": self.state.solution_history,
            "diversity_score": self.state.diversity_score,
            "top_k_score_nodes": self.state.top_k_score_nodes[:dynamic_k], # Dynamic K
            "current_seed_set": self.state.current_seed_set,
            "boundary_info": {
                "neighbor_ids": self.state.neighbor_community_ids,
                "boundary_node_count": len(self.state.boundary_nodes)
            },
            "parameters": {
                "cr1": self.state.cr1,
                "cr2": self.state.cr2,
                "beta": self.state.beta,
                "alpha": self.state.alpha
            },
            "stagnation_count": self.state.stagnation_count,
            "global_dpadv": global_dpadv
        }

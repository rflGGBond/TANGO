"""TANGO Main Entry Point.

TANGO: Topology-Adaptive Neighbor-Governed Orchestration.
Transforms LLM agents from Advisors to Decision-Makers in cooperative coevolution.

Usage:
    python tango/run.py --graphs congress-Twitter --total_budget 20 110 200
"""

import os
import sys
import time
import argparse
import random
import numpy as np

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tango.agents.community_bidding_agent import CommunityBiddingAgent
from tango.agents.negotiation_coordinator import NegotiationCoordinator
from tango.agents.strategic_reviewer import StrategicReviewer
from tango.communication.topology_graph import TopologyAdaptiveGraph
from tango.communication.manager import CommunicationManager
from tango.types import (
    QueryType, CommunityObservation, CommunityAction,
    NegotiationBid, NegotiationResult, NegotiationConflict,
)

# TANGO self-contained imports (from HMACE-derived core/environment)
from tango.environment.env import PCMCCEnvironment
from tango.utils.llm_client import LLMClient
from tango.core.evaluator import DPADVEvaluator
from tango.utils.select_SN import select_SN


DIRECTED_GRAPHS = {"email-Eu-core", "p2p-Gnutella31", "soc-Epinions1",
                    "congress-Twitter", "soc-advogato"}
UNDIRECTED_GRAPHS = {"BA3000", "ER3000", "WS3000"}


def main():
    parser = argparse.ArgumentParser(description="Run TANGO")
    
    # Experiment args
    parser.add_argument("--graphs", type=str, nargs='+', default=["congress-Twitter"])
    parser.add_argument("--total_budget", type=int, nargs='+', default=[20, 110, 200])
    parser.add_argument("--num_communities", type=int, default=16)
    parser.add_argument("--max_gen", type=int, default=20)
    parser.add_argument("--t_comm", type=int, default=5, help="Evolution communication interval")
    parser.add_argument("--t_nego", type=int, default=10, help="Negotiation cycle interval")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mc_runs", type=int, default=10000)
    
    # Danger thresholds
    parser.add_argument("--tau_1", type=float, default=0.3)
    parser.add_argument("--tau_2", type=float, default=0.6)
    
    # LLM args
    parser.add_argument("--llm_provider", type=str, default="local")
    parser.add_argument("--llm_model", type=str, default="Qwen2.5-7B-Instruct")
    parser.add_argument("--model_root", type=str, default="../../models")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    
    # TANGO specific args
    parser.add_argument("--max_nego_rounds", type=int, default=3)
    parser.add_argument("--topology_threshold", type=float, default=0.1)
    parser.add_argument("--max_queries", type=int, default=5)
    
    # Ablation
    parser.add_argument("--disable_negotiation", action="store_true")
    parser.add_argument("--disable_coordinator", action="store_true")
    parser.add_argument("--disable_reviewer", action="store_true")
    parser.add_argument("--disable_communication", action="store_true")
    parser.add_argument("--disable_iterative_revision", action="store_true",
                        help="Disable iterative bid revision (Gap 3 ablation)")
    
    args = parser.parse_args()
    
    # Seed
    random.seed(42)
    np.random.seed(42)
    
    # Print config
    print("=" * 60)
    print("TANGO: Topology-Adaptive Neighbor-Governed Orchestration")
    print("=" * 60)
    print(f"Graphs: {args.graphs}")
    print(f"Budgets: {args.total_budget}")
    print(f"LLM: {args.llm_model} ({args.llm_provider})")
    print(f"Negotiation: {'DISABLED' if args.disable_negotiation else f'Every {args.t_nego} gens'}")
    print(f"Communication: {'DISABLED' if args.disable_communication else 'NR-CIQ'}")
    print(f"Iterative Revision: {'DISABLED' if args.disable_iterative_revision else f'Up to {args.max_nego_rounds} rounds'}")
    print("=" * 60)
    
    # Init LLM
    llm_client = LLMClient(
        provider=args.llm_provider,
        model=args.llm_model,
        model_root=args.model_root,
        api_key=args.api_key,
        base_url=args.base_url,
    )
    
    for graph_name in args.graphs:
        print(f"\n{'#'*50}")
        print(f"Processing: {graph_name}")
        print(f"{'#'*50}")
        
        is_directed = graph_name in DIRECTED_GRAPHS
        graph_path = f"../graph/{graph_name}.txt"
        
        try:
            sn_nodes = select_SN(graph_name, 50, is_directed=is_directed)
        except Exception as e:
            print(f"Error selecting SN: {e}")
            continue
        
        for k in args.total_budget:
            print(f"\n--- Budget k={k} ---")
            
            for r in range(args.repeats):
                print(f"  Repeat {r+1}/{args.repeats}")
                
                # Init PCMCC environment
                env = PCMCCEnvironment(
                    graph_path, sn_nodes, k, args.num_communities,
                    is_directed=is_directed,
                    tau_1=args.tau_1, tau_2=args.tau_2,
                )
                
                # Init TANGO components
                topology = TopologyAdaptiveGraph(
                    connection_threshold=args.topology_threshold)
                comm_manager = CommunicationManager(topology)
                
                cba_agents = {}
                for com_id in env.communities:
                    cba_agents[com_id] = CommunityBiddingAgent(
                        agent_id=f"CBA_{com_id}",
                        llm_client=llm_client,
                        tau_1=args.tau_1, tau_2=args.tau_2,
                    )
                
                coordinator = NegotiationCoordinator(
                    llm_client=llm_client,
                    max_negotiation_rounds=args.max_nego_rounds,
                    total_budget=k,
                )
                
                reviewer = StrategicReviewer(llm_client=llm_client)
                
                # Evolution loop
                start_time = time.time()
                gen = 1
                
                while True:
                    # 1. Standard evolution step
                    env.step(agent_active=False)
                    
                    # 2. Agent interaction (every T_comm)
                    if gen % args.t_comm == 0:
                        # Rebuild communication topology
                        comm_graph = comm_manager.build_topology(
                            env.G, env.communities)
                        
                        # Sync agents with environment
                        current_ids = set(env.communities.keys())
                        for cid in list(cba_agents.keys()):
                            if cid not in current_ids:
                                del cba_agents[cid]
                        for cid in current_ids:
                            if cid not in cba_agents:
                                cba_agents[cid] = CommunityBiddingAgent(
                                    f"CBA_{cid}", llm_client,
                                    tau_1=args.tau_1, tau_2=args.tau_2)
                        
                        # Negotiation cycle
                        if gen % args.t_nego == 0 and not args.disable_negotiation:
                            print(f"\n>>> Negotiation Cycle (gen {gen})")
                            _run_negotiation_cycle(
                                env, cba_agents, comm_manager, 
                                coordinator, reviewer,
                                disable_coordinator=args.disable_coordinator,
                                disable_reviewer=args.disable_reviewer,
                                disable_communication=args.disable_communication,
                                disable_iterative_revision=args.disable_iterative_revision,
                                max_nego_rounds=args.max_nego_rounds,
                            )
                        else:
                            # Standard agent actions
                            for com_id, agent in cba_agents.items():
                                obs_dict = env.communities[com_id].get_observation(
                                    gen, "exploration", env.global_best_dpadv)
                                
                                # Get neighbor states via NR-CIQ
                                if not args.disable_communication:
                                    neighbor_states = comm_manager.get_neighbor_states(
                                        com_id, 
                                        lambda cid, qt: _community_data_provider(
                                            cid, qt, env))
                                else:
                                    neighbor_states = {}
                                
                                obs_dict["neighbor_states"] = neighbor_states
                                obs = CommunityObservation(**{k: v for k, v in obs_dict.items() 
                                                              if k in CommunityObservation.__dataclass_fields__})
                                
                                action = agent.get_action(obs, in_negotiation=False)
                                env.apply_community_action(com_id, action)
                    
                    if env.check_termination(args.max_gen):
                        break
                    
                    gen += 1
                
                elapsed = time.time() - start_time
                print(f"  Done in {elapsed:.1f}s. Best DPADV: {env.global_best_dpadv:.4f}")
                
                # Final evaluation
                if env.global_best_seed:
                    score = DPADVEvaluator.get_activated_node_count(
                        env.global_best_seed, env.G, env.sn_nodes, 
                        runs=args.mc_runs, model='COICM')
                    print(f"  Negative Activated (COICM): {score:.2f}")


# ────────────────────────────────────────────────────────────
# Negotiation Cycle with Iterative Revision (Gap 3)
# ────────────────────────────────────────────────────────────

def _run_negotiation_cycle(env, cba_agents, comm_manager,
                           coordinator, reviewer,
                           disable_coordinator=False,
                           disable_reviewer=False,
                           disable_communication=False,
                           disable_iterative_revision=False,
                           max_nego_rounds=3):
    """
    Execute one full negotiation cycle with iterative bid revision.
    
    Algorithm (Gap 3 – Iterative Refinement):
    ┌─────────────────────────────────────────────────────────┐
    │ PHASE 1: Initial Query & Bid                            │
    │   CBA queries neighbors [EVOLUTION_STATE, BOUNDARY_NODES,│
    │                         BUDGET_PROPOSAL]                │
    │   → generates initial bid with neighbor data            │
    ├─────────────────────────────────────────────────────────┤
    │ PHASE 2: Conflict Detection                             │
    │   Coordinator detects node_overlap & budget conflicts   │
    ├─────────────────────────────────────────────────────────┤
    │ PHASE 3: Iterative Revision (if conflicts & not disabled)│
    │   FOR round in 1..max_nego_rounds:                      │
    │     a. CBAs in conflict query INFLUENCE_ESTIMATE        │
    │     b. CBAs generate revised bids (marginal benefit)    │
    │     c. Coordinator re-evaluates conflicts               │
    │     d. If no conflicts remain → BREAK                   │
    ├─────────────────────────────────────────────────────────┤
    │ PHASE 4: Final Resolution & Review                      │
    │   Coordinator produces final allocation                 │
    │   Reviewer validates (3-level)                          │
    └─────────────────────────────────────────────────────────┘
    """
    
    # ════════════════════════════════════════════════════════
    # PHASE 1: Initial Query & Bid
    # ════════════════════════════════════════════════════════
    bids = []
    previous_bids: Dict[int, NegotiationBid] = {}  # community_id -> previous bid
    
    for com_id, agent in cba_agents.items():
        com = env.communities[com_id]
        obs_dict = com.get_observation(env.current_gen, "negotiation", 
                                       env.global_best_dpadv)
        
        if not disable_communication:
            # Phase 1 query: get evolution state + boundary nodes + budget info
            neighbor_states = comm_manager.get_neighbor_states(
                com_id,
                lambda cid, qt: _community_data_provider(cid, qt, env))
        else:
            neighbor_states = {}
        
        obs_dict["neighbor_states"] = neighbor_states
        obs = CommunityObservation(**{k: v for k, v in obs_dict.items()
                                      if k in CommunityObservation.__dataclass_fields__})
        
        action = agent.get_action(obs, in_negotiation=True)
        
        if action.negotiation_bid:
            action.negotiation_bid.round = 0  # Initial bid
            bids.append(action.negotiation_bid)
            previous_bids[com_id] = action.negotiation_bid
    
    if not bids:
        print("  No bids generated, skipping negotiation")
        return
    
    print(f"  Phase 1: {len(bids)} initial bids generated")
    
    # ════════════════════════════════════════════════════════
    # PHASE 2: Conflict Detection
    # ════════════════════════════════════════════════════════
    if not disable_coordinator:
        conflicts = coordinator.detect_conflicts(bids)
        print(f"  Phase 2: {len(conflicts)} conflict(s) detected")
    else:
        conflicts = []
    
    # ════════════════════════════════════════════════════════
    # PHASE 3: Iterative Revision (Gap 3)
    # ════════════════════════════════════════════════════════
    if conflicts and not disable_iterative_revision and not disable_communication:
        print(f"  Phase 3: Starting iterative revision (max {max_nego_rounds} rounds)...")
        
        for nego_round in range(1, max_nego_rounds + 1):
            print(f"    Revision round {nego_round}/{max_nego_rounds}")
            
            # 3a. Identify communities involved in conflicts
            involved_communities = set()
            for conflict in conflicts:
                involved_communities.add(conflict.community_a)
                involved_communities.add(conflict.community_b)
            # Remove 0 (budget pool)
            involved_communities.discard(0)
            
            # 3b. Query INFLUENCE_ESTIMATE for contested nodes
            influence_responses: Dict[int, Dict[int, Dict[str, Any]]] = {}
            # influence_responses[community_id][neighbor_id] = {expected_gain, ...}
            
            for com_id in involved_communities:
                if com_id not in cba_agents:
                    continue
                
                com = env.communities[com_id]
                obs_dict = com.get_observation(env.current_gen, "negotiation",
                                               env.global_best_dpadv)
                
                # Query neighbors with INFLUENCE_ESTIMATE
                if not disable_communication:
                    # Get neighbors from topology
                    neighbors = comm_manager.get_neighbors(com_id)
                    
                    influence_data = {}
                    for neighbor_id in neighbors:
                        try:
                            data = _community_data_provider(neighbor_id, QueryType.INFLUENCE_ESTIMATE, env)
                            if data:
                                influence_data[neighbor_id] = data
                        except Exception as e:
                            print(f"      Influence query {com_id}→{neighbor_id} failed: {e}")
                    
                    influence_responses[com_id] = influence_data
                    
                    # Also get standard neighbor states for context
                    neighbor_states = comm_manager.get_neighbor_states(
                        com_id,
                        lambda cid, qt: _community_data_provider(cid, qt, env))
                else:
                    neighbor_states = {}
                    influence_responses[com_id] = {}
                
                obs_dict["neighbor_states"] = neighbor_states
                obs = CommunityObservation(**{k: v for k, v in obs_dict.items()
                                              if k in CommunityObservation.__dataclass_fields__})
                
                # 3c. Generate revised bid
                prev_bid = previous_bids.get(com_id)
                if prev_bid is None:
                    continue
                
                # Filter conflicts relevant to this agent
                my_conflicts = [
                    c for c in conflicts
                    if com_id in (c.community_a, c.community_b)
                ]
                
                revised_action = agent._generate_revised_bid(
                    obs, my_conflicts, prev_bid,
                    influence_responses.get(com_id, {}))
                
                if revised_action.negotiation_bid:
                    revised_action.negotiation_bid.round = nego_round
                    # Update bids list: replace old bid for this community
                    bids = [b for b in bids if b.community_id != com_id]
                    bids.append(revised_action.negotiation_bid)
                    previous_bids[com_id] = revised_action.negotiation_bid
                    
                    if revised_action.counter_proposal:
                        pass  # Counter-proposals are tracked in the bid
            
            # 3d. Re-detect conflicts with revised bids
            if not disable_coordinator:
                conflicts = coordinator.detect_conflicts(bids)
                print(f"      Conflicts remaining: {len(conflicts)}")
                
                if not conflicts:
                    print(f"    All conflicts resolved in round {nego_round}!")
                    break
            else:
                conflicts = []
                break
        
        if conflicts:
            print(f"    {len(conflicts)} conflict(s) remain after {max_nego_rounds} rounds")
    
    # ════════════════════════════════════════════════════════
    # PHASE 4: Final Resolution & Review
    # ════════════════════════════════════════════════════════
    
    # Coordinator produces final resolution
    if not disable_coordinator:
        result = coordinator.coordinate(bids, env.communities, env.global_best_dpadv)
    else:
        # No coordinator: just use bids directly
        result = NegotiationResult(
            round=0, consensus_reached=True,
            boundary_allocations={b.community_id: b.boundary_node_proposals for b in bids},
            budget_allocations={b.community_id: b.budget_request for b in bids},
        )
    
    # Reviewer validates
    if not disable_reviewer:
        review_result = reviewer.review(
            result, env.communities, env.G, env.sn_nodes,
            env.global_best_dpadv, DPADVEvaluator)
        
        if review_result.passed:
            print(f"  Phase 4: Negotiation PASSED review ({review_result.stage})")
            _apply_negotiation_result(env, result)
        else:
            print(f"  Phase 4: Negotiation REJECTED: {review_result.rejection_reason}")
    else:
        _apply_negotiation_result(env, result)


def _community_data_provider(community_id: int, query_type, env) -> Dict[str, Any]:
    """
    Provide community data in response to an NR-CIQ query.
    
    Maps QueryType to the appropriate community observation fields.
    This is the callback used by CommunicationManager.route_query().
    """
    try:
        com = env.communities[community_id]
        state = com.state
        obs = com.get_observation(env.current_gen, "negotiation", env.global_best_dpadv)
    except Exception:
        return {}
    
    from tango.utils.types import QueryType as QT
    
    if query_type == QT.EVOLUTION_STATE or str(query_type) == "evolution_state":
        return {
            "current_dpadv": state.current_dpadv,
            "dpadv_history": getattr(state, 'dpadv_history', [])[-10:],
            "diversity_score": getattr(state, 'diversity_score', 0.0),
            "stagnation_count": getattr(state, 'stagnation_count', 0),
            "danger_score": getattr(state, 'danger_score', 0.0),
            "parameters": getattr(state, 'parameters', {}),
            "improvement_rate": getattr(state, 'improvement_rate', 0.0),
            "gamma": getattr(state, 'gamma', 0.0),
            "best_dpadv": state.current_dpadv,
            "budget": state.budget,
            "boundary_node_count": len(getattr(state, 'boundary_nodes', [])),
        }
    
    elif query_type == QT.BOUNDARY_NODES or str(query_type) == "boundary_nodes":
        boundary_nodes = getattr(state, 'boundary_nodes', [])
        return {
            "boundary_nodes": boundary_nodes,
            "boundary_node_count": len(boundary_nodes),
            "neighbor_ids": getattr(state, 'neighbor_community_ids', []),
            "boundary_scores": getattr(state, 'boundary_scores', {}),
            "covered_frontier": getattr(state, 'covered_frontier', []),
            "expected_gain": getattr(state, 'expected_gain', 0.0),
            "current_dpadv": state.current_dpadv,
            "danger_score": getattr(state, 'danger_score', 0.0),
            "budget": state.budget,
        }
    
    elif query_type == QT.TOP_K_CANDIDATES or str(query_type) == "top_k_candidates":
        return {
            "top_k_score_nodes": obs.get("top_k_score_nodes", []) if isinstance(obs, dict) else [],
            "current_seed_set": getattr(state, 'current_seed_set', []),
            "solution_history": getattr(state, 'solution_history', [])[-5:],
        }
    
    elif query_type == QT.BUDGET_PROPOSAL or str(query_type) == "budget_proposal":
        return {
            "budget": state.budget,
            "current_dpadv": state.current_dpadv,
            "boundary_risk": getattr(state, 'boundary_risk', 0.0),
            "improvement_rate": getattr(state, 'improvement_rate', 0.0),
            "marginal_benefit": getattr(state, 'marginal_benefit', 0.0),
            "danger_score": getattr(state, 'danger_score', 0.0),
        }
    
    elif query_type == QT.INFLUENCE_ESTIMATE or str(query_type) == "influence_estimate":
        # INFLUENCE_ESTIMATE: return data needed for marginal benefit comparison
        return {
            "expected_gain": getattr(state, 'expected_gain', 0.0),
            "covered_frontier": getattr(state, 'covered_frontier', []),
            "marginal_benefit": getattr(state, 'marginal_benefit', 0.0),
            "propagation_overlap": getattr(state, 'propagation_overlap', 0.0),
            "current_dpadv": state.current_dpadv,
            "danger_score": getattr(state, 'danger_score', 0.0),
            "budget": state.budget,
        }
    
    # Fallback: return basic state info
    return {
        "current_dpadv": state.current_dpadv,
        "danger_score": getattr(state, 'danger_score', 0.0),
        "budget": state.budget,
    }


def _apply_negotiation_result(env, result):
    """Apply negotiated boundary allocations to the environment."""
    for cid, nodes in result.boundary_allocations.items():
        if cid in env.communities:
            com = env.communities[cid]
            # Merge allocated boundary nodes into community seed set
            current_seeds = set(com.state.current_seed_set)
            for node in nodes:
                if node not in current_seeds:
                    current_seeds.add(node)
            # Trim to budget
            new_seeds = list(current_seeds)[:com.state.budget]
            com.update_best_solution(new_seeds, com.state.current_dpadv)
    
    if result.budget_allocations:
        for cid, budget in result.budget_allocations.items():
            if cid in env.communities:
                env.communities[cid].state.budget = budget


if __name__ == "__main__":
    main()

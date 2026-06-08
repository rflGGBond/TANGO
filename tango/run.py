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
from tango.types import QueryType

# Import HMACE components (to be installed or symlinked)
try:
    from HMACE.environment.env import PCMCCEnvironment
    from HMACE.utils.llm_client import LLMClient
    from HMACE.core.evaluator import DPADVEvaluator
    from HMACE.utils.select_SN import select_SN
except ImportError:
    print("Warning: HMACE not found in path. Please install or symlink HMACE.")


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
                
                # Init HMACE environment
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
                                        lambda cid, qt: env.communities[cid].get_observation(
                                            gen, "exploration", env.global_best_dpadv))
                                else:
                                    neighbor_states = {}
                                
                                from tango.types import CommunityObservation
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


def _run_negotiation_cycle(env, cba_agents, comm_manager,
                           coordinator, reviewer,
                           disable_coordinator=False,
                           disable_reviewer=False,
                           disable_communication=False):
    """Execute one full negotiation cycle."""
    
    bids = []
    
    # Phase 1: Each CBA gets neighbor info and generates bid
    for com_id, agent in cba_agents.items():
        com = env.communities[com_id]
        obs_dict = com.get_observation(env.current_gen, "negotiation", 
                                       env.global_best_dpadv)
        
        if not disable_communication:
            neighbor_states = comm_manager.get_neighbor_states(
                com_id,
                lambda cid, qt: env.communities[cid].get_observation(
                    env.current_gen, "negotiation", env.global_best_dpadv))
        else:
            neighbor_states = {}
        
        obs_dict["neighbor_states"] = neighbor_states
        from tango.types import CommunityObservation
        obs = CommunityObservation(**{k: v for k, v in obs_dict.items()
                                      if k in CommunityObservation.__dataclass_fields__})
        
        action = agent.get_action(obs, in_negotiation=True)
        
        if action.negotiation_bid:
            bids.append(action.negotiation_bid)
    
    if not bids:
        print("  No bids generated, skipping negotiation")
        return
    
    # Phase 2: Coordinator resolves conflicts
    if not disable_coordinator:
        result = coordinator.coordinate(bids, env.communities, env.global_best_dpadv)
    else:
        # No coordinator: just use bids directly
        from tango.types import NegotiationResult
        result = NegotiationResult(
            round=0, consensus_reached=True,
            boundary_allocations={b.community_id: b.boundary_node_proposals for b in bids},
            budget_allocations={b.community_id: b.budget_request for b in bids},
        )
    
    # Phase 3: Reviewer validates
    if not disable_reviewer:
        review_result = reviewer.review(
            result, env.communities, env.G, env.sn_nodes,
            env.global_best_dpadv, DPADVEvaluator)
        
        if review_result.passed:
            print(f"  Negotiation PASSED review ({review_result.stage})")
            _apply_negotiation_result(env, result)
        else:
            print(f"  Negotiation REJECTED: {review_result.rejection_reason}")
    else:
        _apply_negotiation_result(env, result)


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

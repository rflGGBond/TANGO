"""Community Bidding Agent (CBA): P2P negotiating agent for TANGO-CIQ."""

import json
import dataclasses
from typing import Dict, List, Any, Optional

from .base import BaseAgent
from ..utils.types import (
    CommunityObservation, CommunityAction,
    NegotiationBid, CounterProposal, NegotiationConflict
)
from ..utils.llm_client import LLMClient


class CommunityBiddingAgent(BaseAgent):
    """
    Agent controlling a single community with negotiation capabilities.
    
    Extends HMACE's LocalAgent with:
    1. NR-CIQ queries to neighbors for state sharing
    2. Bid generation for boundary node negotiation
    3. Counter-proposal generation when conflicts arise
    
    The agent optimizes for GLOBAL benefit, not just its own community.
    """
    
    def __init__(self, agent_id: str, llm_client: LLMClient = None,
                 tau_1: float = 0.3, tau_2: float = 0.6):
        super().__init__(agent_id, llm_client)
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        
        # Negotiation state
        self._negotiation_history: List[Dict[str, Any]] = []
        
    def get_action(self, observation: CommunityObservation,
                   in_negotiation: bool = False) -> CommunityAction:
        """
        Get action from agent.
        
        Args:
            observation: Current community state including neighbor info
            in_negotiation: Whether this is a negotiation cycle
        """
        if in_negotiation:
            return self._generate_bid(observation)
        else:
            return self._standard_action(observation)
    
    def _standard_action(self, obs: CommunityObservation) -> CommunityAction:
        """Standard evolutionary action (parameter tuning or seed proposal)."""
        obs_dict = dataclasses.asdict(obs)
        obs_dict["solution_history"] = obs_dict.get("solution_history", [])[-5:]
        
        danger_score = obs_dict.get("danger_score", 0.0)
        danger_level = self._get_danger_level(danger_score)
        
        system_prompt = self._build_standard_prompt(obs, danger_level)
        user_prompt = f"Observation: {json.dumps(obs_dict, default=str)}\n\nDecide action type and generate content. Respond with valid JSON only."
        
        try:
            response = self.llm_client.get_completion(system_prompt, user_prompt, temperature=0.7)
            data = json.loads(response)
            
            action = CommunityAction()
            if data.get("action_type") == "adjust_parameters":
                action.parameters = data.get("parameters")
            elif data.get("action_type") == "propose_candidate":
                candidates = data.get("candidate_seed_set", [])
                if candidates and len(candidates) > obs.budget:
                    candidates = candidates[:obs.budget]
                action.candidate_seed_set = candidates
            
            return action
        except Exception as e:
            print(f"CBA {self.agent_id} standard action error: {e}")
            return CommunityAction()
    
    def _generate_bid(self, obs: CommunityObservation) -> CommunityAction:
        """
        Generate a negotiation bid based on own state + neighbor states.
        
        The bid includes:
        - boundary_node_proposals: nodes this agent claims
        - budget_request: requested budget allocation
        - justification: LLM reasoning (for interpretability)
        """
        neighbor_info = obs.neighbor_states
        danger_score = obs.danger_score
        danger_level = self._get_danger_level(danger_score)
        
        # Build negotiation context
        neighbor_summary = self._summarize_neighbors(neighbor_info)
        
        system_prompt = f"""
        You are a Community Bidding Agent in the TANGO-CIQ multi-agent framework.
        
        YOUR GOAL: Maximize GLOBAL blocking effectiveness, not just your own community's score.
        
        COMMUNITY STATE:
        - ID: {obs.community_id}
        - Budget: {obs.budget}
        - Current DPADV: {obs.current_dpadv:.4f}
        - Danger Score: {danger_score:.3f} (Level: {danger_level})
        - Diversity: {obs.diversity_score:.3f}
        - Stagnation: {obs.stagnation_count} generations
        - Boundary Nodes: {obs.boundary_info.get('boundary_node_count', 0)}
        
        NEIGHBOR STATES:
        {neighbor_summary}
        
        NEGOTIATION TASK:
        Propose boundary node allocations and budget adjustments.
        
        1. **Boundary Node Proposal**: Which boundary nodes should your community manage?
           - Consider: which nodes have higher centrality in YOUR community vs neighbor's?
           - If your stagnation is higher, consider conceding nodes to get help from neighbors.
           - If neighbor is in danger, consider taking more responsibility.
        
        2. **Budget Request**: How much budget does your community need?
           - Higher danger → may need more budget for exploration
           - Lower stagnation + high diversity → may be able to share budget
        
        OUTPUT FORMAT (JSON only):
        {{
            "action_type": "negotiate",
            "boundary_node_proposals": [node_id1, node_id2, ...],
            "budget_request": <int>,
            "justification": "Why this allocation benefits global optimization"
        }}
        """
        
        obs_dict = dataclasses.asdict(obs)
        user_prompt = f"Generate negotiation bid. Respond with valid JSON only."
        
        try:
            response = self.llm_client.get_completion(system_prompt, user_prompt, temperature=0.6)
            data = json.loads(response)
            
            bid = NegotiationBid(
                agent_id=self.agent_id,
                community_id=obs.community_id,
                round=0,  # Will be set by Coordinator
                boundary_node_proposals=data.get("boundary_node_proposals", []),
                budget_request=data.get("budget_request", obs.budget),
                justification=data.get("justification", ""),
                neighbor_state_summary=neighbor_info,
            )
            
            self._negotiation_history.append({
                "type": "bid",
                "data": data,
            })
            
            action = CommunityAction(negotiation_bid=bid)
            return action
            
        except Exception as e:
            print(f"CBA {self.agent_id} bid generation error: {e}")
            return CommunityAction()
    
    def generate_counter_proposal(self, conflict: NegotiationConflict,
                                   obs: CommunityObservation) -> CounterProposal:
        """
        Generate a counter-proposal when a conflict is detected.
        
        The agent must decide which contested nodes to concede and which to retain,
        always optimizing for global benefit.
        """
        neighbor_info = obs.neighbor_states
        other_party = (conflict.community_a if conflict.community_a != obs.community_id 
                      else conflict.community_b)
        
        other_state = neighbor_info.get(other_party, {})
        own_danger = obs.danger_score
        other_danger = other_state.get("danger_score", 0.0)
        
        system_prompt = f"""
        You are a Community Bidding Agent resolving a negotiation conflict.
        
        CONFLICT: Community {conflict.community_a} and {conflict.community_b} both claim:
        Nodes: {conflict.contested_nodes}
        
        YOUR COMMUNITY ({obs.community_id}):
        - Danger Score: {own_danger:.3f}
        - Stagnation: {obs.stagnation_count}
        
        OTHER COMMUNITY ({other_party}):
        - Danger Score: {other_danger:.3f}
        - Best DPADV: {other_state.get('best_dpadv', 'N/A')}
        
        DECISION PRINCIPLE: Optimize for GLOBAL benefit.
        - If the OTHER community has HIGHER danger → concede more nodes to help them
        - If YOUR community has HIGHER danger → retain key boundary nodes
        - Consider which community has better centrality for each contested node
        
        OUTPUT FORMAT (JSON only):
        {{
            "concessions": [node_ids_you_give_up],
            "retained_nodes": [node_ids_you_keep],
            "revised_budget": <int>,
            "reasoning": "Why this resolution benefits global optimization"
        }}
        """
        
        try:
            response = self.llm_client.get_completion(system_prompt, 
                f"Context: {conflict.contested_nodes}", temperature=0.5)
            data = json.loads(response)
            
            return CounterProposal(
                agent_id=self.agent_id,
                community_id=obs.community_id,
                original_bid_id="",
                concessions=data.get("concessions", []),
                retained_nodes=data.get("retained_nodes", []),
                revised_budget=data.get("revised_budget", obs.budget),
                reasoning=data.get("reasoning", ""),
            )
        except Exception as e:
            print(f"CBA {self.agent_id} counter-proposal error: {e}")
            return CounterProposal(
                agent_id=self.agent_id,
                community_id=obs.community_id,
                original_bid_id="",
                concessions=conflict.contested_nodes[:len(conflict.contested_nodes)//2],
                retained_nodes=conflict.contested_nodes[len(conflict.contested_nodes)//2:],
                revised_budget=obs.budget,
                reasoning="Fallback: equal split due to LLM error",
            )
    
    def _get_danger_level(self, score: float) -> int:
        if score >= self.tau_2:
            return 2
        elif score >= self.tau_1:
            return 1
        return 0
    
    def _summarize_neighbors(self, neighbor_states: Dict[int, Dict[str, Any]]) -> str:
        """Format neighbor states for LLM prompt."""
        if not neighbor_states:
            return "No neighbor information available."
        
        lines = []
        for nid, state in neighbor_states.items():
            lines.append(
                f"  Neighbor {nid}: DPADV={state.get('current_dpadv', 'N/A'):.4f} "
                f"Danger={state.get('danger_score', 'N/A'):.3f} "
                f"Budget={state.get('budget', 'N/A')} "
                f"Boundary Nodes={state.get('boundary_node_count', 'N/A')}"
            )
        return "\n".join(lines) if lines else "No neighbor information available."
    
    def _build_standard_prompt(self, obs: CommunityObservation, danger_level: int) -> str:
        """Build prompt for standard (non-negotiation) action."""
        danger_note = ""
        if danger_level == 1:
            danger_note = "WARNING: Mild danger detected. Consider more aggressive exploration."
        elif danger_level == 2:
            danger_note = "CRITICAL: Severe danger. Prioritize boundary node injection."
        
        return f"""
        You are Community Bidding Agent for community {obs.community_id}.
        Goal: Minimize DPADV (block negative influence).
        
        Budget: {obs.budget}. Current DPADV: {obs.current_dpadv:.4f}
        Diversity: {obs.diversity_score:.3f}. Stagnation: {obs.stagnation_count}
        
        {danger_note}
        
        Decide action type:
        - "adjust_parameters": Tune cr1, cr2, beta, alpha
        - "propose_candidate": Select seed set of size {obs.budget}
        
        Output JSON only:
        {{"action_type": "...", "parameters": {{...}} OR "candidate_seed_set": [...]}}
        """

"""Community Bidding Agent (CBA): P2P negotiating agent for TANGO."""

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
    4. Marginal benefit analysis using influence estimates
    
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
        
        The bid now incorporates marginal benefit analysis:
        - expected_gain: estimated DPADV reduction per boundary node
        - marginal_benefit: estimated DPADV reduction per additional budget unit
        - covered_frontier: boundary nodes already covered by this community
        - propagation_overlap: Jaccard of influence reachable sets with neighbor
        """
        neighbor_info = obs.neighbor_states
        danger_score = obs.danger_score
        danger_level = self._get_danger_level(danger_score)
        
        # Extract marginal benefit data from neighbor queries
        influence_data = self._extract_influence_data(neighbor_info)
        
        # Build negotiation context with influence metrics
        neighbor_summary = self._summarize_neighbors(neighbor_info)
        
        system_prompt = f"""
        You are a Community Bidding Agent in the TANGO multi-agent framework.
        
        YOUR GOAL: Maximize GLOBAL blocking effectiveness (minimize total DPADV).
        
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
        
        INFLUENCE ESTIMATES (from NR-CIQ queries):
        {json.dumps(influence_data, indent=2) if influence_data else "No influence data available yet."}
        
        NEGOTIATION TASK:
        Propose boundary node allocations and budget adjustments using marginal benefit reasoning.
        
        1. **Marginal Benefit Analysis**:
           - For each boundary node, estimate: ΔDPADV if YOUR community claims it vs neighbor claims it
           - Consider expected_gain: higher expected_gain → stronger claim
           - Consider marginal_benefit: if your marginal_benefit per budget is LOWER than neighbor's,
             consider transferring budget to them (global optimum)
        
        2. **Boundary Node Proposal**: Which boundary nodes should your community manage?
           - Nodes where YOUR expected_gain > neighbor's expected_gain → claim
           - Nodes where neighbor's expected_gain is higher → consider conceding
           - If propagation_overlap is high (>0.5), coordinate to avoid redundant coverage
        
        3. **Budget Request**: How much budget does your community need?
           - Higher danger + high marginal_benefit → request more budget
           - Lower marginal_benefit than neighbors → consider sharing budget
           - If covered_frontier already has many nodes, may need less budget
        
        DECISION PRINCIPLES (in priority order):
        1. Maximize global marginal benefit (assign nodes to community with highest expected_gain)
        2. Address danger asymmetry (high-danger community gets priority on contested nodes)
        3. Avoid redundant coverage (when propagation_overlap is high)
        
        OUTPUT FORMAT (JSON only):
        {{
            "action_type": "negotiate",
            "boundary_node_proposals": [node_id1, node_id2, ...],
            "budget_request": <int>,
            "marginal_benefit_estimate": <float>,
            "expected_gain_per_node": {{"node_id": gain, ...}},
            "justification": "Why this allocation benefits global optimization, citing marginal benefit data"
        }}
        """
        
        try:
            response = self.llm_client.get_completion(system_prompt, 
                "Generate negotiation bid with marginal benefit analysis. Respond with valid JSON only.",
                temperature=0.6)
            data = json.loads(response)
            
            # Extract marginal benefit metrics
            marginal_benefit = data.get("marginal_benefit_estimate", 0.0)
            expected_gain_per_node = data.get("expected_gain_per_node", {})
            # Convert string keys to int if needed
            expected_gain_per_node = {
                int(k) if isinstance(k, str) else k: v 
                for k, v in expected_gain_per_node.items()
            }
            
            # Build propagation overlap from neighbor data
            propagation_overlap = {}
            for nid, nstate in neighbor_info.items():
                if "propagation_overlap" in nstate:
                    propagation_overlap[nid] = nstate["propagation_overlap"]
            
            bid = NegotiationBid(
                agent_id=self.agent_id,
                community_id=obs.community_id,
                round=0,  # Will be set by Coordinator
                boundary_node_proposals=data.get("boundary_node_proposals", []),
                budget_request=data.get("budget_request", obs.budget),
                justification=data.get("justification", ""),
                neighbor_state_summary=neighbor_info,
                marginal_benefit=marginal_benefit,
                expected_gain_per_node=expected_gain_per_node,
                propagation_overlap=propagation_overlap,
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
    
    def _generate_revised_bid(self, obs: CommunityObservation,
                              conflicts: List[NegotiationConflict],
                              previous_bid: NegotiationBid,
                              influence_responses: Dict[int, Dict[str, Any]]) -> CommunityAction:
        """
        Generate a REVISED bid after querying neighbors with INFLUENCE_ESTIMATE.
        
        This is the iterative refinement step (Gap 3):
        1. Query neighbors for INFLUENCE_ESTIMATE data
        2. Re-evaluate marginal benefit with fresh influence data
        3. Propose concessions or stronger claims based on evidence
        
        Args:
            obs: Current community observation
            conflicts: List of conflicts involving this community
            previous_bid: The bid from the previous round
            influence_responses: Fresh INFLUENCE_ESTIMATE data from neighbors
        """
        danger_score = obs.danger_score
        danger_level = self._get_danger_level(danger_score)
        
        # Summarize fresh influence data
        influence_summary = []
        for nid, data in influence_responses.items():
            influence_summary.append({
                "neighbor": nid,
                "expected_gain": data.get("expected_gain", "N/A"),
                "covered_frontier": data.get("covered_frontier", []),
                "marginal_benefit": data.get("marginal_benefit", "N/A"),
                "propagation_overlap": data.get("propagation_overlap", "N/A"),
            })
        
        # Identify contested nodes relevant to this agent
        my_contested = []
        for conflict in conflicts:
            if obs.community_id in (conflict.community_a, conflict.community_b):
                my_contested.extend(conflict.contested_nodes)
        my_contested = list(set(my_contested))
        
        system_prompt = f"""
        You are a Community Bidding Agent REVISING your negotiation bid.
        
        PREVIOUS BID:
        - Proposed Nodes: {previous_bid.boundary_node_proposals}
        - Budget Request: {previous_bid.budget_request}
        - Marginal Benefit: {previous_bid.marginal_benefit:.4f}
        
        CONTESTED NODES (in conflict): {my_contested}
        
        YOUR COMMUNITY ({obs.community_id}):
        - Danger: {danger_score:.3f} (Level: {danger_level})
        - Stagnation: {obs.stagnation_count}
        - Current DPADV: {obs.current_dpadv:.4f}
        
        FRESH INFLUENCE ESTIMATES FROM NEIGHBORS:
        {json.dumps(influence_summary, indent=2)}
        
        REVISION TASK:
        Re-evaluate your bid using the fresh influence estimates.
        
        1. **For each contested node**, compare expected_gain:
           - If YOUR expected_gain > neighbor's expected_gain → retain the node, strengthen your claim
           - If neighbor's expected_gain > YOUR expected_gain → CONCEDE the node (this benefits global optimum)
        
        2. **Budget revision**: 
           - If your marginal_benefit < neighbor's marginal_benefit → reduce budget, let neighbor use it
           - If your marginal_benefit > neighbor's → maintain or increase budget request
        
        3. **Propagation overlap**:
           - High overlap (>0.7) → coordinate: let ONE community handle the frontier, avoid double-coverage
           - Low overlap (<0.3) → both communities should act independently
        
        OUTPUT FORMAT (JSON only):
        {{
            "action_type": "revise_bid",
            "concessions": [node_ids_to_give_up],
            "retained_nodes": [node_ids_still_claimed],
            "new_claims": [node_ids_newly_claimed],
            "revised_budget": <int>,
            "revised_marginal_benefit": <float>,
            "reasoning": "Explain concessions/retentions based on marginal benefit comparison"
        }}
        """
        
        try:
            response = self.llm_client.get_completion(system_prompt,
                f"Revise bid based on influence data. Contested nodes: {my_contested}",
                temperature=0.5)
            data = json.loads(response)
            
            # Build revised bid
            retained = data.get("retained_nodes", [])
            new_claims = data.get("new_claims", [])
            
            revised_bid = NegotiationBid(
                agent_id=self.agent_id,
                community_id=obs.community_id,
                round=previous_bid.round + 1,
                boundary_node_proposals=list(set(retained + new_claims)),
                budget_request=data.get("revised_budget", obs.budget),
                justification=data.get("reasoning", ""),
                neighbor_state_summary=obs.neighbor_states,
                marginal_benefit=data.get("revised_marginal_benefit", 0.0),
            )
            
            # Also build counter-proposal for concessions tracking
            counter = CounterProposal(
                agent_id=self.agent_id,
                community_id=obs.community_id,
                original_bid_id="",
                concessions=data.get("concessions", []),
                retained_nodes=retained,
                revised_budget=data.get("revised_budget", obs.budget),
                reasoning=data.get("reasoning", ""),
            )
            
            self._negotiation_history.append({
                "type": "revised_bid",
                "data": data,
            })
            
            action = CommunityAction(
                negotiation_bid=revised_bid,
                counter_proposal=counter,
            )
            return action
            
        except Exception as e:
            print(f"CBA {self.agent_id} revised bid error: {e}")
            # Fallback: keep previous bid
            return CommunityAction(negotiation_bid=previous_bid)
    
    def generate_counter_proposal(self, conflict: NegotiationConflict,
                                   obs: CommunityObservation) -> CounterProposal:
        """
        Generate a counter-proposal when a conflict is detected.
        
        Now incorporates influence estimate data for evidence-based concessions.
        """
        neighbor_info = obs.neighbor_states
        other_party = (conflict.community_a if conflict.community_a != obs.community_id 
                      else conflict.community_b)
        
        other_state = neighbor_info.get(other_party, {})
        own_danger = obs.danger_score
        other_danger = other_state.get("danger_score", 0.0)
        
        # Extract influence data for contested nodes
        own_expected_gains = other_state.get("expected_gain", {})
        if isinstance(own_expected_gains, (int, float)):
            own_expected_gains = {}
        other_expected_gains = other_state.get("expected_gain", {})
        if isinstance(other_expected_gains, (int, float)):
            other_expected_gains = {}
        
        # Build per-node comparison
        node_comparisons = []
        for node in conflict.contested_nodes:
            own_gain = own_expected_gains.get(str(node), own_expected_gains.get(node, "unknown"))
            other_gain = other_expected_gains.get(str(node), other_expected_gains.get(node, "unknown"))
            node_comparisons.append({
                "node": node,
                "my_expected_gain": own_gain,
                "their_expected_gain": other_gain,
            })
        
        system_prompt = f"""
        You are a Community Bidding Agent resolving a negotiation conflict.
        
        CONFLICT: Community {conflict.community_a} and {conflict.community_b} both claim nodes.
        
        YOUR COMMUNITY ({obs.community_id}):
        - Danger Score: {own_danger:.3f}
        - Stagnation: {obs.stagnation_count}
        - Marginal Benefit: {obs.neighbor_states.get(obs.community_id, {}).get('marginal_benefit', 'N/A')}
        
        OTHER COMMUNITY ({other_party}):
        - Danger Score: {other_danger:.3f}
        - Best DPADV: {other_state.get('best_dpadv', 'N/A')}
        - Marginal Benefit: {other_state.get('marginal_benefit', 'N/A')}
        
        PER-NODE EXPECTED GAIN COMPARISON:
        {json.dumps(node_comparisons, indent=2)}
        
        DECISION PRINCIPLES (in priority order):
        1. **Marginal gain superiority**: If other has HIGHER expected_gain for a node → CONCEDE it
        2. **Danger priority**: If danger scores differ significantly, higher danger gets priority
        3. **Stagnation relief**: If one community is stagnant and the other is improving, the stagnant one may need more resources
        
        OUTPUT FORMAT (JSON only):
        {{
            "concessions": [node_ids_you_give_up],
            "retained_nodes": [node_ids_you_keep],
            "revised_budget": <int>,
            "marginal_gain_conceded": <float>,
            "marginal_gain_retained": <float>,
            "reasoning": "Per-node justification based on expected_gain comparison"
        }}
        """
        
        try:
            response = self.llm_client.get_completion(system_prompt, 
                f"Context: contested nodes {conflict.contested_nodes}", temperature=0.5)
            data = json.loads(response)
            
            return CounterProposal(
                agent_id=self.agent_id,
                community_id=obs.community_id,
                original_bid_id="",
                concessions=data.get("concessions", []),
                retained_nodes=data.get("retained_nodes", []),
                revised_budget=data.get("revised_budget", obs.budget),
                reasoning=data.get("reasoning", ""),
                marginal_gain_conceded=data.get("marginal_gain_conceded", 0.0),
                marginal_gain_retained=data.get("marginal_gain_retained", 0.0),
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
    
    # ── Helper methods ──────────────────────────────────────────
    
    def _get_danger_level(self, score: float) -> int:
        if score >= self.tau_2:
            return 2
        elif score >= self.tau_1:
            return 1
        return 0
    
    def _summarize_neighbors(self, neighbor_states: Dict[int, Dict[str, Any]]) -> str:
        """Format neighbor states for LLM prompt, including influence data."""
        if not neighbor_states:
            return "No neighbor information available."
        
        lines = []
        for nid, state in neighbor_states.items():
            dpadv = state.get('current_dpadv', 'N/A')
            dpadv_str = f"{dpadv:.4f}" if isinstance(dpadv, (int, float)) else str(dpadv)
            danger = state.get('danger_score', 'N/A')
            danger_str = f"{danger:.3f}" if isinstance(danger, (int, float)) else str(danger)
            
            line = (
                f"  Neighbor {nid}: DPADV={dpadv_str} "
                f"Danger={danger_str} "
                f"Budget={state.get('budget', 'N/A')} "
                f"Boundary Nodes={state.get('boundary_node_count', 'N/A')}"
            )
            
            # Add influence data when available
            extra = []
            if 'expected_gain' in state:
                eg = state['expected_gain']
                extra.append(f"ExpectedGain={eg:.4f}" if isinstance(eg, float) else f"ExpectedGain={eg}")
            if 'marginal_benefit' in state:
                mb = state['marginal_benefit']
                extra.append(f"MarginalBenefit={mb:.4f}" if isinstance(mb, float) else f"MarginalBenefit={mb}")
            if 'propagation_overlap' in state:
                po = state['propagation_overlap']
                extra.append(f"PropOverlap={po:.3f}" if isinstance(po, float) else f"PropOverlap={po}")
            if 'covered_frontier' in state:
                cf = state['covered_frontier']
                if isinstance(cf, list):
                    extra.append(f"CoveredFrontier={len(cf)} nodes")
            
            if extra:
                line += " [" + ", ".join(extra) + "]"
            
            lines.append(line)
        
        return "\n".join(lines) if lines else "No neighbor information available."
    
    def _extract_influence_data(self, neighbor_states: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Extract influence estimate fields from neighbor responses."""
        result = {}
        for nid, state in neighbor_states.items():
            entry = {}
            for field in ["expected_gain", "covered_frontier", "marginal_benefit", "propagation_overlap"]:
                if field in state:
                    entry[field] = state[field]
            if entry:
                result[str(nid)] = entry
        return result
    
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

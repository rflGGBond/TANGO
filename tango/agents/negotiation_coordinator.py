"""Negotiation Coordinator: Global resource optimization and conflict arbitration."""

import json
from typing import Dict, List, Tuple, Optional

from .base import BaseAgent
from ..utils.types import (
    NegotiationBid, NegotiationConflict, NegotiationResult,
    CounterProposal, GlobalObservation, GlobalAction
)
from ..utils.llm_client import LLMClient


class NegotiationCoordinator(BaseAgent):
    """
    Coordinates the P2P negotiation process:
    1. Aggregates all Community Bidding Agent bids
    2. Detects conflicts (overlapping node claims, budget oversubscription)
    3. Facilitates counter-proposals for conflict resolution
    4. Produces final boundary allocation and budget distribution
    
    KEY DIFFERENCE from HMACE's GlobalAgent:
    - Coordinator does NOT make merge decisions
    - Coordinator works WITH agent proposals, not instead of them
    - Allocates resources based on negotiated inputs, not top-down commands
    """
    
    def __init__(self, llm_client: LLMClient = None,
                 max_negotiation_rounds: int = 3,
                 total_budget: int = 0):
        super().__init__("NegotiationCoordinator", llm_client)
        self.max_rounds = max_negotiation_rounds
        self.total_budget = total_budget
    
    def detect_conflicts(self, bids: List[NegotiationBid]) -> List[NegotiationConflict]:
        """Detect conflicts among agent bids."""
        conflicts = []
        
        # 1. Node overlap conflicts
        node_claims: Dict[int, List[int]] = {}  # node_id -> [community_ids claiming it]
        for bid in bids:
            for node in bid.boundary_node_proposals:
                if node not in node_claims:
                    node_claims[node] = []
                node_claims[node].append(bid.community_id)
        
        # Find nodes claimed by multiple communities
        contested = {n: cs for n, cs in node_claims.items() if len(cs) > 1}
        processed_pairs = set()
        
        for node, communities in contested.items():
            for i, cid_a in enumerate(communities):
                for cid_b in communities[i+1:]:
                    pair = tuple(sorted([cid_a, cid_b]))
                    if pair not in processed_pairs:
                        # Collect all contested nodes between this pair
                        pair_nodes = [n for n, cs in contested.items() 
                                     if cid_a in cs and cid_b in cs]
                        conflicts.append(NegotiationConflict(
                            community_a=cid_a,
                            community_b=cid_b,
                            contested_nodes=pair_nodes,
                            conflict_type="node_overlap"
                        ))
                        processed_pairs.add(pair)
        
        # 2. Budget oversubscription
        total_requested = sum(b.budget_request for b in bids)
        if total_requested > self.total_budget:
            # Find the communities with the largest requests
            sorted_bids = sorted(bids, key=lambda b: b.budget_request, reverse=True)
            for bid in sorted_bids[:2]:
                conflicts.append(NegotiationConflict(
                    community_a=bid.community_id,
                    community_b=0,  # 0 = global budget pool
                    contested_nodes=[],
                    conflict_type="budget_oversubscription"
                ))
        
        return conflicts
    
    def coordinate(self, bids: List[NegotiationBid],
                   communities: Dict[int, any],
                   global_dpadv: float) -> NegotiationResult:
        """
        Run a full coordination cycle:
        1. Detect conflicts
        2. For each conflict, request counter-proposals
        3. Resolve and produce final allocation
        """
        all_conflicts = self.detect_conflicts(bids)
        
        if not all_conflicts:
            # No conflicts - all bids are compatible
            return NegotiationResult(
                round=0,
                consensus_reached=True,
                boundary_allocations={b.community_id: b.boundary_node_proposals 
                                     for b in bids},
                budget_allocations={b.community_id: b.budget_request for b in bids},
            )
        
        # Use LLM to resolve conflicts
        resolved = self._llm_resolve(bids, all_conflicts, global_dpadv)
        return resolved
    
    def _llm_resolve(self, bids: List[NegotiationBid],
                     conflicts: List[NegotiationConflict],
                     global_dpadv: float) -> NegotiationResult:
        """Use LLM to resolve conflicts and produce final allocation."""
        
        bid_summaries = []
        for b in bids:
            bid_summaries.append({
                "community": b.community_id,
                "claimed_nodes": b.boundary_node_proposals,
                "budget_request": b.budget_request,
                "justification": b.justification,
            })
        
        conflict_summaries = []
        for c in conflicts:
            conflict_summaries.append({
                "between": [c.community_a, c.community_b],
                "contested_nodes": c.contested_nodes,
                "type": c.conflict_type,
            })
        
        system_prompt = f"""
        You are the Negotiation Coordinator in TANGO.
        
        GLOBAL STATE:
        - Total Budget: {self.total_budget}
        - Current Global DPADV: {global_dpadv:.4f}
        
        BIDS RECEIVED:
        {json.dumps(bid_summaries, indent=2)}
        
        CONFLICTS DETECTED:
        {json.dumps(conflict_summaries, indent=2)}
        
        TASK: Resolve all conflicts and produce final boundary node and budget allocations.
        
        PRINCIPLES:
        1. Optimize for GLOBAL blocking effectiveness
        2. Contested boundary nodes → assign to community with stronger local centrality
        3. Budget conflicts → prioritize communities with higher danger/stagnation
        4. If a community already has many allocated nodes, it may need less budget
        
        OUTPUT FORMAT (JSON only):
        {{
            "boundary_allocations": {{"community_id": [node_ids], ...}},
            "budget_allocations": {{"community_id": budget, ...}},
            "reasoning": "Concise explanation of resolution decisions"
        }}
        
        CONSTRAINTS:
        - Sum of budget_allocations MUST equal {self.total_budget}
        - Each boundary node MUST be assigned to exactly ONE community
        """
        
        try:
            response = self.llm_client.get_completion(system_prompt,
                "Resolve conflicts. Respond with valid JSON only.", temperature=0.4)
            data = json.loads(response)
            
            boundary = {int(k): v for k, v in data.get("boundary_allocations", {}).items()}
            budget = {int(k): v for k, v in data.get("budget_allocations", {}).items()}
            
            # Enforce budget constraint
            total = sum(budget.values())
            if total != self.total_budget and budget:
                first_key = next(iter(budget))
                budget[first_key] += (self.total_budget - total)
            
            return NegotiationResult(
                round=1,
                consensus_reached=True,
                boundary_allocations=boundary,
                budget_allocations=budget,
            )
            
        except Exception as e:
            print(f"Coordinator LLM error: {e}")
            # Fallback: proportional allocation
            return self._fallback_resolve(bids)
    
    def _fallback_resolve(self, bids: List[NegotiationBid]) -> NegotiationResult:
        """Fallback: equal allocation when LLM fails."""
        n = len(bids)
        budget_per = self.total_budget // n if n > 0 else 0
        
        return NegotiationResult(
            round=1,
            consensus_reached=True,
            boundary_allocations={b.community_id: b.boundary_node_proposals for b in bids},
            budget_allocations={b.community_id: budget_per for b in bids},
        )

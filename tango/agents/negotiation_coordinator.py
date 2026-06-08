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
    - Uses marginal benefit analysis for evidence-based conflict resolution
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
        
        Now uses marginal_benefit and expected_gain_per_node
        for evidence-based conflict resolution.
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
        
        # Use LLM to resolve conflicts with marginal benefit data
        resolved = self._llm_resolve(bids, all_conflicts, global_dpadv)
        return resolved
    
    def _llm_resolve(self, bids: List[NegotiationBid],
                     conflicts: List[NegotiationConflict],
                     global_dpadv: float) -> NegotiationResult:
        """Use LLM to resolve conflicts and produce final allocation.
        
        Now incorporates marginal_benefit, expected_gain_per_node,
        and propagation_overlap from bids for evidence-based decisions.
        """
        
        bid_summaries = []
        for b in bids:
            summary = {
                "community": b.community_id,
                "claimed_nodes": b.boundary_node_proposals,
                "budget_request": b.budget_request,
                "justification": b.justification,
            }
            # Include marginal benefit data when available
            if b.marginal_benefit:
                summary["marginal_benefit"] = round(b.marginal_benefit, 6)
            if b.expected_gain_per_node:
                # Summarize: top 5 nodes by expected gain
                top_gains = sorted(b.expected_gain_per_node.items(), 
                                   key=lambda x: x[1], reverse=True)[:5]
                summary["top_expected_gains"] = [
                    {"node": n, "gain": round(g, 6)} for n, g in top_gains
                ]
            if b.propagation_overlap:
                summary["propagation_overlap_with"] = {
                    str(nid): round(ov, 4) for nid, ov in b.propagation_overlap.items()
                }
            bid_summaries.append(summary)
        
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
        
        BIDS RECEIVED (with marginal benefit analysis):
        {json.dumps(bid_summaries, indent=2)}
        
        CONFLICTS DETECTED:
        {json.dumps(conflict_summaries, indent=2)}
        
        TASK: Resolve all conflicts and produce final boundary node and budget allocations.
        
        EVIDENCE-BASED RESOLUTION PRINCIPLES:
        1. **Marginal Gain Priority**: For each contested node, compare expected_gain between claimants.
           Assign the node to the community with HIGHER expected_gain (better global outcome).
        2. **Propagation Overlap**: If two communities have high propagation_overlap (>0.5),
           let only ONE handle the shared frontier to avoid redundant coverage.
        3. **Budget Efficiency**: If community A's marginal_benefit > community B's,
           allocate more budget to A (higher DPADV reduction per budget unit).
        4. **Danger/Stagnation Tiebreaker**: If marginal gains are similar, prioritize the community
           with higher danger_score or stagnation_count.
        
        OUTPUT FORMAT (JSON only):
        {{
            "boundary_allocations": {{"community_id": [node_ids], ...}},
            "budget_allocations": {{"community_id": budget, ...}},
            "reasoning": "Per-conflict explanation citing marginal benefit evidence"
        }}
        
        CONSTRAINTS:
        - Sum of budget_allocations MUST equal {self.total_budget}
        - Each boundary node MUST be assigned to exactly ONE community
        - If marginal_benefit data is available, use it as primary decision signal
        """
        
        try:
            response = self.llm_client.get_completion(system_prompt,
                "Resolve conflicts using marginal benefit evidence. Respond with valid JSON only.", 
                temperature=0.4)
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
            # Fallback: proportional allocation using marginal benefit weights
            return self._fallback_resolve(bids)
    
    def _fallback_resolve(self, bids: List[NegotiationBid]) -> NegotiationResult:
        """Fallback: marginal-benefit-weighted allocation when LLM fails."""
        n = len(bids)
        if n == 0:
            return NegotiationResult(round=1, consensus_reached=True,
                                     boundary_allocations={}, budget_allocations={})
        
        # Use marginal_benefit as weight if available, else equal split
        marginal_benefits = []
        for b in bids:
            mb = b.marginal_benefit if b.marginal_benefit > 0 else 1.0 / n
            marginal_benefits.append(mb)
        
        total_mb = sum(marginal_benefits)
        weights = [mb / total_mb for mb in marginal_benefits]
        
        budget_alloc = {}
        remaining = self.total_budget
        for i, b in enumerate(bids):
            alloc = max(1, int(self.total_budget * weights[i]))
            budget_alloc[b.community_id] = alloc
            remaining -= alloc
        
        # Distribute remainder to highest marginal benefit
        if remaining > 0 and bids:
            best_idx = marginal_benefits.index(max(marginal_benefits))
            budget_alloc[bids[best_idx].community_id] += remaining
        
        # For node allocations, use expected_gain_per_node to break ties
        boundary_alloc = {}
        for b in bids:
            if b.expected_gain_per_node:
                # Sort proposed nodes by expected gain, keep top nodes
                sorted_nodes = sorted(b.expected_gain_per_node.items(),
                                      key=lambda x: x[1], reverse=True)
                boundary_alloc[b.community_id] = [n for n, _ in sorted_nodes]
            else:
                boundary_alloc[b.community_id] = b.boundary_node_proposals
        
        return NegotiationResult(
            round=1,
            consensus_reached=True,
            boundary_allocations=boundary_alloc,
            budget_allocations=budget_alloc,
        )

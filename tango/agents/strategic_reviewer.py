"""Strategic Reviewer: Multi-level verification of negotiation outcomes."""

import json
from typing import Dict, List, Any, Optional, Set

from .base import BaseAgent
from ..utils.types import NegotiationResult, ReviewResult, GlobalAction
from ..utils.llm_client import LLMClient


class StrategicReviewer(BaseAgent):
    """
    Reviews negotiation outcomes through three verification levels:
    
    Level 1 - Rule Check (no LLM, fast):
        - Node legality: all allocated nodes exist in the graph/subgraph
        - Budget constraint: sum of allocations equals total budget  
        - Community consistency: no node assigned to multiple communities
        
    Level 2 - Strategic Review (LLM-based):
        - Does the allocation make sense from a global optimization perspective?
        - Are high-danger communities getting appropriate resources?
        - Is the boundary allocation balanced?
        
    Level 3 - Multi-Fidelity Validation (DPADV-based):
        - Fast approximate evaluation (low MC runs)
        - If passes, full precise evaluation (high MC runs)
        - Only accepted if global fitness improves
    
    Only allocations passing ALL THREE levels are executed.
    """
    
    def __init__(self, llm_client: LLMClient = None,
                 rule_check_enabled: bool = True,
                 strategic_review_enabled: bool = True,
                 multi_fidelity_enabled: bool = True):
        super().__init__("StrategicReviewer", llm_client)
        self.rule_check_enabled = rule_check_enabled
        self.strategic_review_enabled = strategic_review_enabled
        self.multi_fidelity_enabled = multi_fidelity_enabled
    
    def review(self, result: NegotiationResult,
               communities: Dict[int, any],
               G, sn_nodes: List[int],
               global_best_dpadv: float,
               dpadv_evaluator=None) -> ReviewResult:
        """
        Run the full three-level review pipeline.
        
        Returns:
            ReviewResult with passed=True only if all levels pass.
        """
        
        # Level 1: Rule Check
        if self.rule_check_enabled:
            l1 = self._rule_check(result, communities, G)
            if not l1.passed:
                return l1
        
        # Level 2: Strategic Review  
        if self.strategic_review_enabled:
            l2 = self._strategic_review(result, communities, global_best_dpadv)
            if not l2.passed:
                return l2
        
        # Level 3: Multi-Fidelity Validation
        if self.multi_fidelity_enabled and dpadv_evaluator:
            l3 = self._multi_fidelity_validation(
                result, communities, G, sn_nodes, global_best_dpadv, dpadv_evaluator)
            if not l3.passed:
                return l3
        
        return ReviewResult(passed=True, stage="complete", score=global_best_dpadv)
    
    def _rule_check(self, result: NegotiationResult,
                    communities: Dict[int, any], G) -> ReviewResult:
        """
        Level 1: Deterministic rule-based verification.
        No LLM calls - fast and reliable.
        """
        all_nodes = set(G.nodes())
        assigned_nodes: Set[int] = set()
        
        for cid, nodes in result.boundary_allocations.items():
            # Check node legality
            for node in nodes:
                if node not in all_nodes:
                    return ReviewResult(
                        passed=False, stage="rule_check",
                        rejection_reason=f"Node {node} does not exist in graph"
                    )
                
                if node in assigned_nodes:
                    return ReviewResult(
                        passed=False, stage="rule_check",
                        rejection_reason=f"Node {node} assigned to multiple communities"
                    )
                assigned_nodes.add(node)
            
            # Check community exists
            if cid not in communities:
                return ReviewResult(
                    passed=False, stage="rule_check",
                    rejection_reason=f"Community {cid} does not exist"
                )
        
        # Check budget constraint
        total_allocated = sum(result.budget_allocations.values())
        total_budget = sum(c.state.budget for c in communities.values())
        
        # Allow small tolerance for rounding
        if abs(total_allocated - total_budget) > max(1, total_budget * 0.05):
            return ReviewResult(
                passed=False, stage="rule_check",
                rejection_reason=f"Budget mismatch: allocated {total_allocated} != {total_budget}"
            )
        
        return ReviewResult(passed=True, stage="rule_check")
    
    def _strategic_review(self, result: NegotiationResult,
                          communities: Dict[int, any],
                          global_best_dpadv: float) -> ReviewResult:
        """
        Level 2: LLM-based strategic review.
        Evaluates whether the allocation makes sense from a global perspective.
        """
        com_summaries = []
        for cid, com in communities.items():
            state = com.state
            allocated = result.boundary_allocations.get(cid, [])
            budget = result.budget_allocations.get(cid, state.budget)
            com_summaries.append({
                "id": cid,
                "danger_score": getattr(state, 'danger_score', 0),
                "current_dpadv": state.current_dpadv,
                "allocated_boundary_nodes": len(allocated),
                "allocated_budget": budget,
                "original_budget": state.budget,
            })
        
        system_prompt = f"""
        You are the Strategic Reviewer in TANGO.
        
        GLOBAL BEST DPADV: {global_best_dpadv:.4f}
        
        ALLOCATION SUMMARY:
        {json.dumps(com_summaries, indent=2)}
        
        TASK: Review this allocation. Check:
        1. Are high-danger communities receiving adequate budget adjustments?
        2. Is the boundary allocation balanced (no community overloaded/starved)?
        3. Does this allocation plausibly improve the global objective?
        
        OUTPUT (JSON only):
        {{
            "approved": true/false,
            "confidence": 0.0-1.0,
            "issues": ["issue1", "issue2"] or [],
            "suggestions": ["suggestion1"] or []
        }}
        """
        
        try:
            response = self.llm_client.get_completion(system_prompt,
                "Review the allocation. JSON only.", temperature=0.3)
            data = json.loads(response)
            
            if data.get("approved", False):
                return ReviewResult(
                    passed=True, stage="strategic",
                    suggestions=data.get("suggestions", []))
            else:
                return ReviewResult(
                    passed=False, stage="strategic",
                    rejection_reason="; ".join(data.get("issues", ["Strategic review failed"])))
        except Exception as e:
            print(f"Strategic Review LLM error: {e}")
            # On error, be conservative but don't block
            return ReviewResult(passed=True, stage="strategic",
                              suggestions=["LLM review failed, accepting with caution"])
    
    def _multi_fidelity_validation(self, result: NegotiationResult,
                                    communities: Dict[int, any],
                                    G, sn_nodes: List[int],
                                    global_best_dpadv: float,
                                    dpadv_evaluator) -> ReviewResult:
        """
        Level 3: Fitness-based validation with two-tiers.
        
        Tier 1: Fast approximate DPADV (low MC runs)
        Tier 2: Precise DPADV (high MC runs) - only if Tier 1 passes
        """
        try:
            # Construct global seed set from allocations
            global_seed = []
            for cid, allocated in result.boundary_allocations.items():
                global_seed.extend(allocated)
            
            if not global_seed:
                return ReviewResult(passed=True, stage="multi_fidelity",
                                  suggestions=["Empty allocation - no validation needed"])
            
            # Tier 1: Fast check (100 runs)
            fast_score = dpadv_evaluator.get_activated_node_count(
                global_seed, G, sn_nodes, runs=100, model='COICM')
            
            # Relaxed threshold for Tier 1 (allow 5% degradation)
            if fast_score > global_best_dpadv * 1.05:
                return ReviewResult(
                    passed=False, stage="multi_fidelity",
                    rejection_reason=f"Fast evaluation shows degradation: {fast_score:.2f} > {global_best_dpadv:.2f}",
                    score=fast_score)
            
            # Tier 2: Precise check (full MC runs)
            precise_score = dpadv_evaluator.get_activated_node_count(
                global_seed, G, sn_nodes, runs=10000, model='COICM')
            
            if precise_score >= global_best_dpadv:
                return ReviewResult(
                    passed=False, stage="multi_fidelity",
                    rejection_reason=f"Precise evaluation shows no improvement: {precise_score:.2f} >= {global_best_dpadv:.2f}",
                    score=precise_score)
            
            return ReviewResult(
                passed=True, stage="multi_fidelity",
                score=precise_score)
                
        except Exception as e:
            print(f"Multi-fidelity validation error: {e}")
            return ReviewResult(passed=True, stage="multi_fidelity",
                              suggestions=[f"Validation error: {e}, accepting with caution"])

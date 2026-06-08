"""NR-CIQ: Neighbor-Restricted Coordinated Inquiry Protocol."""

import json
import uuid
from typing import Dict, Any, List, Optional
from ..utils.types import NRCIQQuery, NRCIQResponse, QueryType


ALLOWED_FIELDS = {
    QueryType.EVOLUTION_STATE: [
        "current_dpadv", "dpadv_history", "diversity_score",
        "stagnation_count", "danger_score", "parameters",
        "improvement_rate", "gamma"
    ],
    QueryType.BOUNDARY_NODES: [
        "boundary_nodes", "boundary_node_count", "neighbor_ids",
        "boundary_scores",  # node_id -> centrality score at boundary
        "covered_frontier",  # subset of boundary nodes already covered by this community
        "expected_gain",     # estimated DPADV reduction if this boundary node is claimed
    ],
    QueryType.TOP_K_CANDIDATES: [
        "top_k_score_nodes", "current_seed_set", "solution_history"
    ],
    QueryType.BUDGET_PROPOSAL: [
        "budget", "current_dpadv", "boundary_risk", "improvement_rate",
        "marginal_benefit",  # estimated DPADV reduction per additional budget unit
    ],
    # NEW: Influence-specific query
    QueryType.INFLUENCE_ESTIMATE: [
        "expected_gain",       # estimated DPADV reduction for claiming this node
        "covered_frontier",    # boundary nodes this community already covers
        "marginal_benefit",    # ΔDPADV / Δbudget at current allocation
        "propagation_overlap",  # Jaccard of influence reachable sets
    ],
}


class NRCIQProtocol:
    """Defines and enforces the NR-CIQ structured query protocol."""
    
    @staticmethod
    def create_query(sender_id: int, receiver_id: int, 
                     query_type: QueryType, fields: List[str]) -> NRCIQQuery:
        """Create a structured NR-CIQ query."""
        allowed = ALLOWED_FIELDS.get(query_type, [])
        valid_fields = [f for f in fields if f in allowed]
        
        if not valid_fields:
            raise ValueError(f"No valid fields for query type {query_type}. "
                           f"Allowed: {allowed}")
        
        return NRCIQQuery(
            query_id=str(uuid.uuid4())[:8],
            sender_id=sender_id,
            receiver_id=receiver_id,
            query_type=query_type,
            fields=valid_fields,
        )
    
    @staticmethod
    def create_response(query: NRCIQQuery, sender_id: int,
                        data: Dict[str, Any], status: str = "ok") -> NRCIQResponse:
        """Create a structured response to a query."""
        filtered_data = {k: v for k, v in data.items() if k in query.fields}
        
        return NRCIQResponse(
            query_id=query.query_id,
            sender_id=sender_id,
            receiver_id=query.sender_id,
            data=filtered_data,
            status=status
        )
    
    @staticmethod
    def format_for_llm(response: NRCIQResponse) -> str:
        """Format an NR-CIQ response as JSON for LLM consumption."""
        return json.dumps({
            "from_community": response.sender_id,
            "query_type": response.data.get("query_type", "unknown"),
            "data": response.data
        }, default=str, indent=2)
    
    @staticmethod
    def extract_neighbor_info(responses: List[NRCIQResponse]) -> Dict[int, Dict[str, Any]]:
        """Aggregate responses from multiple neighbors into a single dict."""
        neighbor_states = {}
        for resp in responses:
            if resp.status == "ok":
                neighbor_states[resp.sender_id] = resp.data
        return neighbor_states

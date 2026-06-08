"""Extended types for TANGO negotiation framework."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum


# ============================================================
# NR-CIQ Communication Types
# ============================================================

class QueryType(Enum):
    EVOLUTION_STATE = "evolution_state"
    BOUNDARY_NODES = "boundary_nodes"
    TOP_K_CANDIDATES = "top_k_candidates"
    BUDGET_PROPOSAL = "budget_proposal"
    INFLUENCE_ESTIMATE = "influence_estimate"


@dataclass
class NRCIQQuery:
    """Structured query from one agent to a neighbor."""
    query_id: str
    sender_id: int
    receiver_id: int
    query_type: QueryType
    fields: List[str]  # Specific fields requested
    timestamp: int = 0


@dataclass
class NRCIQResponse:
    """Structured response to an NR-CIQ query."""
    query_id: str
    sender_id: int  # responder
    receiver_id: int  # original querier
    data: Dict[str, Any]
    status: str = "ok"  # ok | denied | partial


# ============================================================
# Negotiation Types
# ============================================================

@dataclass
class NegotiationBid:
    """A bid submitted by a Community Bidding Agent."""
    agent_id: int
    community_id: int
    round: int
    boundary_node_proposals: List[int]  # Nodes this agent claims
    budget_request: int  # Requested budget allocation
    justification: str  # LLM-generated reasoning
    neighbor_state_summary: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    # NEW: Marginal benefit analysis from influence queries
    marginal_benefit: float = 0.0
    expected_gain_per_node: Dict[int, float] = field(default_factory=dict)
    propagation_overlap: Dict[int, float] = field(default_factory=dict)


@dataclass
class NegotiationConflict:
    """Conflict detected by Coordinator between two agents."""
    community_a: int
    community_b: int
    contested_nodes: List[int]
    conflict_type: str  # "node_overlap" | "budget_oversubscription"


@dataclass
class CounterProposal:
    """Agent's response to a conflict."""
    agent_id: int
    community_id: int
    original_bid_id: str
    concessions: List[int]  # Nodes conceded to other party
    retained_nodes: List[int]  # Nodes still claimed
    revised_budget: int
    reasoning: str
    # NEW: Influence-based justification
    marginal_gain_conceded: float = 0.0
    marginal_gain_retained: float = 0.0


# ============================================================
# Negotiation Result
# ============================================================

@dataclass
class NegotiationResult:
    """Final result of a negotiation cycle."""
    round: int
    consensus_reached: bool
    boundary_allocations: Dict[int, List[int]]  # community_id -> allocated boundary nodes
    budget_allocations: Dict[int, int]  # community_id -> budget
    unresolved_conflicts: List[NegotiationConflict] = field(default_factory=list)


# ============================================================
# Reviewer Types
# ============================================================

@dataclass
class ReviewResult:
    """Result of Strategic Reviewer evaluation."""
    passed: bool
    stage: str  # rule_check | strategic | multi_fidelity
    score: float = 0.0
    rejection_reason: str = ""
    suggestions: List[str] = field(default_factory=list)


# ============================================================
# Communication Topology Types
# ============================================================

@dataclass
class CommunicationEdge:
    """Weighted edge in the agent communication graph."""
    source: int
    target: int
    weight: float
    propagation_strength: float = 0.0
    boundary_risk: float = 0.0


# ============================================================
# Extended Community Types (reusing HMACE patterns)
# ============================================================

@dataclass
class CommunityObservation:
    """Observation passed to Community Bidding Agent. Extended from HMACE."""
    community_id: int
    current_generation: int
    budget: int
    current_dpadv: float
    dpadv_history: List[float]
    diversity_score: float
    top_k_score_nodes: List[Dict[str, Any]]
    current_seed_set: List[int]
    boundary_info: Dict[str, Any]  # neighbor_ids, boundary_nodes, boundary_node_count
    parameters: Dict[str, float]
    global_dpadv: float
    stagnation_count: int = 0
    danger_score: float = 0.0
    solution_history: List[Dict[str, Any]] = field(default_factory=list)
    # Extended for TANGO
    neighbor_states: Dict[int, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class CommunityAction:
    """Action from Community Bidding Agent. Extended from HMACE."""
    # Mode A: Parameter Adjustment
    parameters: Optional[Dict[str, float]] = None
    
    # Mode B: Candidate Generation
    candidate_seed_set: Optional[List[int]] = None
    
    # Mode C: Negotiation Bid (NEW for TANGO)
    negotiation_bid: Optional[NegotiationBid] = None
    counter_proposal: Optional[CounterProposal] = None


@dataclass
class CommunitySummary:
    """Summary of a community for Global observation."""
    community_id: int
    budget: int
    best_dpadv: float
    improvement_rate: float
    diversity: float
    boundary_risk: float
    danger_score: float
    gamma: float
    closeness_info: Dict[int, float] = field(default_factory=dict)


@dataclass
class GlobalObservation:
    """Observation passed to Negotiation Coordinator and Strategic Reviewer."""
    current_generation: int
    current_global_dpadv: float
    global_dpadv_history: List[float]
    community_summaries: List[CommunitySummary]
    active_bids: List[NegotiationBid] = field(default_factory=list)
    pending_conflicts: List[NegotiationConflict] = field(default_factory=list)
    merge_history: List[Any] = field(default_factory=list)
    emergency_global_call: bool = False


@dataclass
class GlobalAction:
    """Action from Coordinator / Reviewer."""
    global_baselines: Optional[Dict[str, float]] = None
    budget_adjustments: Optional[Dict[int, int]] = None
    boundary_allocations: Optional[Dict[int, List[int]]] = None
    merge_suggestions: Optional[List[Tuple[int, int]]] = None
    rejected_bids: Optional[List[int]] = None  # agent_ids whose bids were rejected

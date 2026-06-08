"""Communication Manager: Audits, routes, and caches NR-CIQ queries."""

import time
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from .topology_graph import TopologyAdaptiveGraph
from .nr_ciq import NRCIQProtocol
from ..utils.types import NRCIQQuery, NRCIQResponse, QueryType


class CommunicationManager:
    """
    Manages all inter-agent communication:
    1. Audits query legality (is receiver a valid neighbor?)
    2. Routes queries to target communities
    3. Caches responses to avoid redundant queries
    4. Enforces rate limits and query budgets
    """
    
    def __init__(self, 
                 topology_builder: TopologyAdaptiveGraph,
                 cache_ttl: int = 3,  # generations before cache invalidates
                 max_queries_per_cycle: int = 5):
        self.topology = topology_builder
        self.protocol = NRCIQProtocol()
        self.cache_ttl = cache_ttl
        self.max_queries_per_cycle = max_queries_per_cycle
        
        # Communication graph (rebuilt each cycle)
        self.comm_graph = None
        
        # Query cache: (sender, receiver, query_type_hash) -> (response, timestamp)
        self._cache: Dict[Tuple[int, int, int], Tuple[NRCIQResponse, int]] = {}
        
        # Query counter per cycle (reset each cycle)
        self._query_counts: Dict[int, int] = defaultdict(int)
        
    def build_topology(self, G, communities: Dict[int, any]):
        """Rebuild communication topology for current cycle."""
        self.comm_graph = self.topology.build(G, communities)
        self._query_counts.clear()
        return self.comm_graph
    
    def audit_query(self, query: NRCIQQuery) -> bool:
        """
        Audit a query for legality.
        Returns True if the query is allowed, False otherwise.
        """
        # Check 1: Is receiver a neighbor?
        if self.comm_graph is None:
            return False
        
        if not self.comm_graph.has_edge(query.sender_id, query.receiver_id):
            return False
        
        # Check 2: Rate limit
        if self._query_counts[query.sender_id] >= self.max_queries_per_cycle:
            return False
        
        # Check 3: Valid query type
        if not isinstance(query.query_type, QueryType):
            return False
        
        return True
    
    def route_query(self, query: NRCIQQuery, 
                    community_data_provider: callable) -> Optional[NRCIQResponse]:
        """
        Route a query to the target community and get response.
        
        Args:
            query: The NR-CIQ query
            community_data_provider: Function(receiver_id, query_type) -> Dict[str, Any]
        """
        if not self.audit_query(query):
            return self.protocol.create_response(
                query, query.receiver_id, {}, status="denied")
        
        # Check cache
        cache_key = (query.sender_id, query.receiver_id, 
                     hash(frozenset(query.fields)))
        
        current_gen = query.timestamp
        if cache_key in self._cache:
            cached_response, cached_time = self._cache[cache_key]
            if current_gen - cached_time <= self.cache_ttl:
                self._query_counts[query.sender_id] += 1
                return cached_response
        
        # Get data from community
        try:
            data = community_data_provider(query.receiver_id, query.query_type)
            response = self.protocol.create_response(query, query.receiver_id, data)
            
            # Cache the response
            self._cache[cache_key] = (response, current_gen)
            self._query_counts[query.sender_id] += 1
            
            return response
        except Exception as e:
            return self.protocol.create_response(
                query, query.receiver_id, 
                {"error": str(e)}, status="partial")
    
    def query_neighbors(self, agent_id: int, query_types: List[QueryType],
                        community_data_provider: callable) -> List[NRCIQResponse]:
        """
        Convenience method: query all neighbors with given query types.
        """
        if self.comm_graph is None:
            return []
        
        neighbors = list(self.comm_graph.neighbors(agent_id))
        responses = []
        
        for neighbor_id in neighbors:
            for qtype in query_types:
                query = self.protocol.create_query(
                    agent_id, neighbor_id, qtype,
                    fields=self._default_fields(qtype))
                query.timestamp = int(time.time())
                
                response = self.route_query(query, community_data_provider)
                if response:
                    responses.append(response)
        
        return responses
    
    def _default_fields(self, query_type: QueryType) -> List[str]:
        """Get default fields for a query type."""
        from .nr_ciq import ALLOWED_FIELDS
        return ALLOWED_FIELDS.get(query_type, [])[:5]  # Top 5 most relevant
    
    def get_neighbor_states(self, agent_id: int,
                            community_data_provider: callable) -> Dict[int, Dict[str, Any]]:
        """Get aggregated state information from all neighbors."""
        responses = self.query_neighbors(
            agent_id,
            [QueryType.EVOLUTION_STATE, QueryType.BOUNDARY_NODES],
            community_data_provider)
        return self.protocol.extract_neighbor_info(responses)
    
    def get_neighbors(self, community_id: int) -> List[int]:
        """Get communication neighbors for a community (convenience wrapper)."""
        if self.comm_graph is None:
            return []
        return list(self.comm_graph.neighbors(community_id))
    
    def clear_cache(self):

        """Clear the response cache."""
        self._cache.clear()

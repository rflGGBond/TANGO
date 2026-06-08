"""Base agent class for TANGO-CIQ."""

from abc import ABC, abstractmethod
from ..utils.llm_client import LLMClient


class BaseAgent(ABC):
    """Abstract base for all TANGO-CIQ agents."""
    
    def __init__(self, agent_id: str, llm_client: LLMClient = None):
        self.agent_id = agent_id
        self.llm_client = llm_client or LLMClient()

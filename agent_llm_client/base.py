from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple

class BaseLLMClient(ABC):

    @abstractmethod
    async def chat(
        self, 
        messages: List[Dict[str, Any]], 
        tools: List[Dict[str, Any]] | None = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Sends chat messages and returns (raw_response_text, metrics_dict)."""
        pass

    @abstractmethod
    def get_embeddings(self, text: str) -> List[float]:
        """Generates vector embeddings for input text."""
        pass
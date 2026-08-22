from .base import BaseLLMClient
from .ollama import OllamaClient
from .gemini import GemeniClient

def create_llm_client(provider: str = "ollama", api_key: str | None = None, **kwargs) -> BaseLLMClient:
    """Factory function creating LLM client instances."""
    provider_lower = provider.lower()

    if provider_lower == "ollama":
        return OllamaClient(**kwargs)
    elif provider_lower in ("gemini", "google"):
        return GemeniClient(api_key=api_key, **kwargs)
    else:
        raise ValueError(f"Unsupported provider: '{provider}'. Choose 'ollama' or 'gemeni' ")

__all__ = [
    "BaseLLMClient",
    "OllamaClient",
    "GemeniClient",
    "create_llm_client"
]
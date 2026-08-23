from .base import BaseLLMClient
from .providers.ollama import OllamaClient
from .providers.gemini import GemeniClient
from .providers.claude import ClaudeClient
from .providers.openrouter import OpenRouterClient

def create_llm_client(provider: str = "ollama", api_key: str | None = None, **kwargs) -> BaseLLMClient:
    """Factory function creating LLM client instances."""
    provider_lower = provider.lower()

    if provider_lower == "ollama":
        return OllamaClient(**kwargs)
    elif provider_lower in ("gemini", "google"):
        return GemeniClient(api_key=api_key, **kwargs)
    elif provider_lower == "claude":
        return ClaudeClient(api_key=api_key, **kwargs)
    elif provider_lower in ("openrouter", "opencode"):
        return OpenRouterClient(api_key=api_key, **kwargs)
    else:
        raise ValueError(f"Unsupported provider: '{provider}'. Choose 'ollama' or 'gemeni' ")

__all__ = [
    "BaseLLMClient",
    "OllamaClient",
    "GemeniClient",
    "ClaudeClient",
    "OpenRouterClient",
    "create_llm_client"
]
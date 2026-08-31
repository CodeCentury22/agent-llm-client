from .base import BaseLLMClient
from .providers.ollama import OllamaClient
from .providers.gemini import GeminiClient
from .providers.claude import ClaudeClient
from .providers.openrouter import OpenRouterClient
from . import providers  # Exposes submodules for pkgutil / unittest.mock

def create_llm_client(provider: str = "ollama", api_key: str | None = None, **kwargs) -> BaseLLMClient:
    """Factory function creating LLM client instances."""
    provider_lower = provider.lower()

    if provider_lower == "ollama":
        return OllamaClient(**kwargs)
    elif provider_lower in ("gemini", "google"):
        return GeminiClient(api_key=api_key, **kwargs)
    elif provider_lower == "claude":
        return ClaudeClient(api_key=api_key, **kwargs)
    elif provider_lower in ("openrouter", "opencode"):
        return OpenRouterClient(api_key=api_key, **kwargs)
    else:
        raise ValueError(
            f"Unsupported provider: '{provider}'. Choose 'ollama', 'gemini', 'claude', or 'openrouter'."
        )

__all__ = [
    "BaseLLMClient",
    "OllamaClient",
    "GeminiClient",
    "ClaudeClient",
    "OpenRouterClient",
    "create_llm_client",
    "providers",
]
import os
import time
from typing import List, Dict, Any, Tuple
import httpx
from ..base import BaseLLMClient
from .ollama import OllamaClient


class OpenRouterClient(BaseLLMClient):
    """Client wrapper for OpenRouter & OpenCode OpenAI-compatible gateways."""

    def __init__(
        self,
        model: str = "deepseek/deepseek-r1",
        api_key: str | None = None,
        embedding_client: BaseLLMClient | None = None,
        **kwargs
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required for OpenRouter/OpenCode provider.")

        self.base_url = kwargs.get("base_url", "https://openrouter.ai/api/v1/chat/completions")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self._fallback_embedding_client = embedding_client or OllamaClient(**kwargs)

    async def chat(self, messages: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
        start_time = time.time()
        payload = {
            "model": self.model,
            "messages": messages
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(self.base_url, headers=self.headers, json=payload, timeout=60.0)
            response.raise_for_status()
            data = response.json()

        elapsed_sec = round(time.time() - start_time, 4)
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        metrics = {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_duration": elapsed_sec,
            "provider": "openrouter"
        }

        return content, metrics

    def get_embeddings(self, text: str) -> List[float]:
        return self._fallback_embedding_client.get_embeddings(text)
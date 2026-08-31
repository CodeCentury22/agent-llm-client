import os
import time
from typing import List, Dict, Any, Tuple
import httpx
from ..base import BaseLLMClient
from .ollama import OllamaClient


class ClaudeClient(BaseLLMClient):
    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: str | None = None,
        embedding_client: BaseLLMClient | None = None,
        **kwargs
    ):
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key is required for Claude provider.")
        
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        # Fallback to local Ollama for embeddings since Anthropic has no native embedding API
        self._embedding_fallback = embedding_client or OllamaClient(**kwargs)

    async def chat(
        self, 
        messages: List[Dict[str, Any]], 
        tools: List[Dict[str, Any]] | None = None
    ) -> Tuple[str, Dict[str, Any]]:
        start_time = time.time()

        system_prompt = None
        filtered_messages = []

        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content")
            else:
                filtered_messages.append(msg)

        payload: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": filtered_messages
        }
        if system_prompt:
            payload["system"] = system_prompt

        if tools:
            formatted_tools = []
            for tool in tools:
                if "function" in tool:
                    func = tool["function"]
                    formatted_tools.append({
                        "name": func.get("name"),
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {})
                    })
                else:
                    formatted_tools.append(tool)
            payload["tools"] = formatted_tools

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(self.base_url, headers=self.headers, json=payload, timeout=60.0)
                response.raise_for_status()
                data = response.json()

            content = data["content"][0]["text"] if data.get("content") else ""
            usage = data.get("usage", {})
            latency = round(time.time() - start_time, 2)

            metrics = {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_duration_sec": latency,
                "provider": "claude"
            }

            return content, metrics
        except Exception as e:
            print(f"❌ [Claude API Error]: {str(e)}")
            return "{}", {}

    def get_embeddings(self, text: str) -> List[float]:
        return self._embedding_fallback.get_embeddings(text)
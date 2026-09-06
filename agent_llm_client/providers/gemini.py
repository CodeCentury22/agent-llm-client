import os
import time
import re
import asyncio
from typing import List, Dict, Any, Tuple
from ..base import BaseLLMClient
from .ollama import OllamaClient


class GeminiClient(BaseLLMClient):

    def __init__(
            self,
            api_key: str | None = None,
            model: str = "gemini-2.5-flash",
            embedding_client: BaseLLMClient | None = None,
            **kwargs
    ):
        from google import genai

        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "API key is required for gemini provider. "
                "Pass 'api_key' or set 'GEMINI_API_KEY'."
            )
        self.client = genai.Client(api_key=self.api_key)
        self.model = model
        self._embedding_fallback = embedding_client or OllamaClient(**kwargs)

    async def chat(
        self, 
        messages: List[Dict[str, Any]], 
        tools: List[Dict[str, Any]] | None = None
    ) -> Tuple[str, Dict[str, Any]]:
        from google.genai import types
        from google.genai.errors import APIError

        start_time = time.perf_counter()
        contents = []
        system_instruction = None

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                system_instruction = content
            elif role == "user":
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=content)]))
            elif role == "assistant":
                contents.append(types.Content(role="model", parts=[types.Part.from_text(text=content)]))

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.0,
            # Disable SDK-side automatic execution of function callables to suppress the SDK AFC warning
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
        )

        for attempt in range(5):
            try:
                # Use async client variant (.aio) to avoid blocking the event loop
                response = await self.client.aio.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config
                )
                elapsed_sec = round(time.perf_counter() - start_time, 2)
                response_text = response.text.strip() if response.text else "{}"

                metrics = {
                    "input_tokens": response.usage_metadata.prompt_token_count if response.usage_metadata else 0,
                    "output_tokens": response.usage_metadata.candidates_token_count if response.usage_metadata else 0,
                    "total_duration_sec": elapsed_sec,
                    "provider": "gemini"
                }

                json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
                return (json_match.group(0) if json_match else response_text), metrics
            except APIError as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print("\n⚠️ [Gemini 429] Rate limit hit. Retrying in 15 seconds...")
                    await asyncio.sleep(15)
                else:
                    print(f"❌ [Gemini API Error]: {str(e)}")
                    return "{}", {}
            except Exception as e:
                print(f"❌ [Gemini Error]: {str(e)}")
                return "{}", {}

        return "{}", {}

    def get_embeddings(self, text: str) -> List[float]:
            return self._embedding_fallback.get_embeddings(text)
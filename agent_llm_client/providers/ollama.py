import json
import re
import urllib.request
import asyncio
from typing import List, Dict, Any, Tuple
from ..base import BaseLLMClient

class OllamaClient(BaseLLMClient):

    def __init__(
            self,
            model: str = "qwen2.5-coder:7b-instruct",
            embed_model: str = "nomic-embed-text",
            host: str = "http://localhost:11434"
    ):
        self.model = model
        self.embed_model = embed_model
        self.host = host

    async def chat(self, messages: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        url = f"{self.host}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format" : "json",
            "option": {"temperature": 0.0, "num_predict": 4096}
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

        event_loop = asyncio.get_event_loop()

        try:
            raw_bytes = await event_loop.run_in_executor(None, lambda: urllib.request.urlopen(req).read())
            res_json = json.loads(raw_bytes.decode("utf-8"))

            metrics = {
                "input_tokens": res_json.get("prompt_eval_count", 0),
                "output_token": res_json.get("eval_count", 0),
                "total_duration_sec": round(res_json.get("total_duration", 0) / 1e9, 2),
                "provider": "ollama"
            }

            content = res_json.get("message", {}).get("content", "").strip()
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            return (json_match.group(0) if json_match else content), metrics
        except Exception as e:
            print(f"❌ [Ollama Error]: {str(e)}")
            return "{}", {}

    def get_embeddings(self, text) -> List[float]:
        url = f"{self.host}/api/embeddings"
        payload = {"model": self.embed_model, "prompt": text}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            response = urllib.request.urlopen(req)
            res_json = json.loads(response.read().decode("utf-8"))
            return res_json.get("embedding", [])
        except Exception as e:
            print(f"❌ [Ollama Embedding Error]: {str(e)}")
            return []
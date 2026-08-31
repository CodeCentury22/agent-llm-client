import json
import re
import urllib.request
import asyncio
from typing import List, Dict, Any, Tuple
from ..base import BaseLLMClient
from agent_async_runner import execute_async_subprocess

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

    def get_installed_models(self) -> List[str]:
        """Queries local Ollama instance for currently installed models."""
        url = f"{self.host}/api/tags"
        req = urllib.request.Request(url)
        try:
            response = urllib.request.urlopen(req, timeout=2)
            if response.status == 200:
                res_json = json.loads(response.read().decode("utf-8"))
                return [m["name"] for m in res_json.get("models", [])]
        except Exception:
            pass
        return []

    async def ensure_model_available(self, model_name: str | None = None) -> bool:
        """Verifies if a model exists locally, prompting an async pull if missing."""
        target_model = model_name or self.model
        installed = self.get_installed_models()

        # Match exact tag or implicit base model tag prefix
        if any(m == target_model or m.startswith(f"{target_model}:") for m in installed):
            return True

        print(f"\n⚠️  [Ollama Notice]: Model '{target_model}' is not installed locally.")
        confirm = input(f"👉 Would you like to pull '{target_model}' now? (y/N): ").strip().lower()

        if confirm == "y":
            print(f"📥 [Downloading Model]: Executing 'ollama pull {target_model}'...")
            res = await execute_async_subprocess(
                f"ollama pull {target_model}",
                timeout=900.0,
                bypass_hitl=True
            )
            if res.get("status") == "SUCCESS":
                print(f"✅ Successfully downloaded '{target_model}'.")
                return True
            else:
                print(f"❌ Failed to download model: {res.get('stderr')}")
                return False
        return False
    
    async def chat(
        self, 
        messages: List[Dict[str, Any]], 
        tools: List[Dict[str, Any]] | None = None
    ) -> Tuple[str, Dict[str, Any]]:
        url = f"{self.host}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "option": {"temperature": 0.0, "num_predict": 4096}
        }

        if tools:
            payload["tools"] = tools

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

        event_loop = asyncio.get_event_loop()

        try:
            raw_bytes = await event_loop.run_in_executor(None, lambda: urllib.request.urlopen(req).read())
            res_json = json.loads(raw_bytes.decode("utf-8"))

            metrics = {
                "input_tokens": res_json.get("prompt_eval_count", 0),
                "output_tokens": res_json.get("eval_count", 0),
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
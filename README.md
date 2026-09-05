# agent-llm-client 🤖

Unified LLM provider abstraction layer and vector embedding generation for local (Ollama) and cloud (Gemini) models.

## Key Features

* **Unified Client Contract:** Seamlessly switch between local Ollama inference and cloud APIs (Gemini) with zero changes to your agent loop.
* **API Key Guardrails:** Strict runtime checks enforcing API key configuration for external cloud providers while keeping local execution friction-free.
* **Built-in Telemetry:** Standardized token counts, total latency duration, and provider tracking on every completion call.
* **Embeddings Interface:** Generate vector embeddings across local and cloud models for seamless vector memory storage.

## Installation

Add directly to any `uv`-managed project:

```bash
uv add git+https://github.com/CodeCentury22/agent-llm-client.git@v0.4.4
```

## Quick Start

```python
import asyncio

from agent_llm_client import create_llm_client


async def main():
    # Local Ollama client (no API key required)
    client = create_llm_client(
        provider="ollama",
        model="qwen2.5-coder:7b-instruct",
    )

    # Chat completion
    messages = [
        {
            "role": "user",
            "content": "Hello agent!",
        }
    ]

    response, metrics = await client.chat(messages)

    print("Response:", response)
    print("Telemetry:", metrics)

    # Generate vector embeddings
    embeddings = client.get_embeddings(
        "sample codebase text for vector indexing"
    )

    print(f"Generated {len(embeddings)}-dimensional vector.")


asyncio.run(main())
```

## License

MIT

import pytest
import httpx
from unittest.mock import patch, MagicMock, AsyncMock
from agent_llm_client import create_llm_client
from agent_llm_client.providers.ollama import OllamaClient
from agent_llm_client.providers.gemini import GeminiClient
from agent_llm_client.providers.claude import ClaudeClient
from agent_llm_client.providers.openrouter import OpenRouterClient

def tet_factory_ollama_instantiation():
    client = create_llm_client("ollama")
    assert isinstance(client, OllamaClient)

def test_gemini_api_key_raises_value_error():
    with pytest.raises(ValueError, match="API key is required"):
        create_llm_client("gemini", api_key=None)

def test_gemini_with_api_key_instantiates():
    client = create_llm_client("gemini", api_key="dummy_key_for_test")
    assert isinstance(client, GeminiClient)

# ==========================================
# 2. OLLAMA LOCAL PROVIDER TESTS
# ==========================================

@pytest.mark.asyncio
@patch("urllib.request.urlopen")
async def test_ollama_chat_success(mock_urlopen):
    # Mock Ollama API response payload
    mock_response = MagicMock()
    mock_response.read.return_value = b'{"message": {"content": "{\\"tool_name\\": \\"list_files_tool\\", \\"arguments\\": {}}"}, "prompt_eval_count": 10, "eval_count": 5, "total_duration": 1000000000}'
    mock_urlopen.return_value = mock_response

    client = OllamaClient()
    messages = [{"role": "user", "content": "List files"}]
    response_text, metrics = await client.chat(messages=messages)

    assert "list_files_tool" in response_text
    assert metrics["provider"] == "ollama"
    assert metrics["input_tokens"] == 10
    assert metrics["total_duration_sec"] == 1.0

@patch("urllib.request.urlopen")
def test_ollama_get_embeddings_success(mock_urlopen):
    mock_response = MagicMock()
    mock_response.read.return_value = b'{"embedding": [0.1, 0.2, 0.3]}'
    mock_urlopen.return_value = mock_response

    client = OllamaClient()
    embeddings = client.get_embeddings("sample code text")

    assert embeddings == [0.1, 0.2, 0.3]

# ==========================================
# 3. GEMINI CLOUD PROVIDER TESTS
# ==========================================

@pytest.mark.asyncio
@patch("google.genai.Client")
async def test_gemini_chat_success(mock_genai_client):
    # Mock Gemini SDK client response
    mock_response = MagicMock()
    mock_response.text = '{"tool_name": "DONE", "arguments": {}}'
    mock_response.usage_metadata.prompt_token_count = 15
    mock_response.usage_metadata.candidates_token_count = 8

    mock_instance = MagicMock()
    mock_instance.models.generate_content.return_value = mock_response
    mock_genai_client.return_value = mock_instance

    client = GeminiClient(api_key="test_key")
    messages = [{"role": "user", "content": "Complete task"}]
    response_text, metrics = await client.chat(messages)

    assert "DONE" in response_text
    assert metrics["provider"] == "gemini"
    assert metrics["input_tokens"] == 15
    assert metrics["output_tokens"] == 8

@patch("google.genai.Client")
def test_gemini_get_embeddings_success(mock_genai_client):
    mock_embedding = MagicMock()
    mock_embedding.values = [0.5, 0.6, 0.7]

    mock_response = MagicMock()
    mock_response.embeddings = [mock_embedding]

    mock_instance = MagicMock()
    mock_instance.models.embed_content.return_value = mock_response
    mock_genai_client.return_value = mock_instance

    client = GeminiClient(api_key="test_key")
    embeddings = client.get_embeddings("sample code text")

    assert embeddings == [0.5, 0.6, 0.7]


# ==========================================
# 4. CLAUDE PROVIDER TESTS
# ==========================================

@pytest.mark.asyncio
@patch("httpx.AsyncClient.post", new_callable=AsyncMock)
async def test_claude_chat_success(mock_post):
    dummy_req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    mock_post.return_value = httpx.Response(
        status_code=200,
        json={
            "content": [{"text": "Hello from Claude!"}],
            "usage": {"input_tokens": 12, "output_tokens": 8}
        },
        request=dummy_req
    )

    client = ClaudeClient(api_key="test_anthropic_key")
    messages = [{"role": "user", "content": "Hi"}]
    response_text, metrics = await client.chat(messages)

    assert response_text == "Hello from Claude!"
    assert metrics["provider"] == "claude"
    assert metrics["input_tokens"] == 12
    assert metrics["output_tokens"] == 8

@patch("agent_llm_client.providers.claude.OllamaClient")
def test_claude_get_embeddings_fallback(mock_ollama_cls):
    mock_ollama_instance = MagicMock()
    mock_ollama_instance.get_embeddings.return_value = [0.1, 0.2, 0.3]
    mock_ollama_cls.return_value = mock_ollama_instance

    client = ClaudeClient(api_key="test_key")
    embeddings = client.get_embeddings("sample_text")

    assert embeddings == [0.1, 0.2, 0.3]

# ==========================================
# 5. OPENROUTER PROVIDER TESTS
# ==========================================

@pytest.mark.asyncio
@patch("httpx.AsyncClient.post", new_callable=AsyncMock)
async def test_openrouter_chat_success(mock_post):
    dummy_req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")

    mock_post.return_value = httpx.Response(
        status_code=200,
        json={
            "choices": [{"message": {"content": "Hello from OpenRouter!"}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 15}
        },
        request=dummy_req
    )

    client = OpenRouterClient(api_key="test_openrouter_key")
    messages = [{"role": "user", "content": "Hello"}]
    response_text, metrics = await client.chat(messages)

    assert response_text == "Hello from OpenRouter!"
    assert metrics["provider"] == "openrouter"
    assert metrics["input_tokens"] == 20
    assert metrics["output_tokens"] == 15

@patch("agent_llm_client.providers.openrouter.OllamaClient")
def test_openrouter_get_embeddings(mock_ollama_cls):
    mock_ollama_instance = MagicMock()
    mock_ollama_instance.get_embeddings.return_value = [0.4, 0.5, 0.6]
    mock_ollama_cls.return_value = mock_ollama_instance

    client = OpenRouterClient(api_key="test_openrouter_api_key")
    embeddings = client.get_embeddings("Sample text")

    assert embeddings == [0.4, 0.5, 0.6]


@pytest.fixture
def client():
    return OllamaClient(model="qwen2.5-coder:32b-instruct")


def test_get_installed_models_success(client):
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.read.return_value = b'{"models": [{"name": "llama3.3:70b"}, {"name": "deepseek-r1:32b"}]}'

    with patch("urllib.request.urlopen", return_value=mock_response):
        models = client.get_installed_models()
        assert models == ["llama3.3:70b", "deepseek-r1:32b"]


def test_get_installed_models_failure(client):
    with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
        models = client.get_installed_models()
        assert models == []


@pytest.mark.asyncio
async def test_ensure_model_available_already_installed(client):
    with patch.object(client, "get_installed_models", return_value=["qwen2.5-coder:32b-instruct"]):
        result = await client.ensure_model_available()
        assert result is True


from unittest.mock import patch, AsyncMock

@pytest.mark.asyncio
async def test_ensure_model_available_pull_success(client):
    with patch.object(client, "get_installed_models", return_value=[]), \
         patch("builtins.input", return_value="y"), \
         patch(
             "agent_llm_client.providers.ollama.execute_async_subprocess",
             new_callable=AsyncMock,
             return_value={"status": "SUCCESS"}
         ) as mock_exec:
        
        result = await client.ensure_model_available("deepseek-r1:32b")
        assert result is True
        mock_exec.assert_called_once_with(
            "ollama pull deepseek-r1:32b",
            timeout=900.0,
            bypass_hitl=True
        )


@pytest.mark.asyncio
async def test_ensure_model_available_pull_declined(client):
    with patch.object(client, "get_installed_models", return_value=[]), \
         patch("builtins.input", return_value="n"):
        
        result = await client.ensure_model_available("deepseek-r1:32b")
        assert result is False
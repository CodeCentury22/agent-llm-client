import pytest
from unittest.mock import patch, MagicMock
from agent_llm_client import create_llm_client, OllamaClient, GemeniClient

def tet_factory_ollama_instantiation():
    client = create_llm_client("ollama")
    assert isinstance(client, OllamaClient)

def test_gemini_api_key_raises_value_error():
    with pytest.raises(ValueError, match="API key is required"):
        create_llm_client("gemini", api_key=None)

def test_gemini_with_api_key_instantiates():
    client = create_llm_client("gemini", api_key="dummy_key_for_test")
    assert isinstance(client, GemeniClient)

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

    client = GemeniClient(api_key="test_key")
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

    client = GemeniClient(api_key="test_key")
    embeddings = client.get_embeddings("sample code text")

    assert embeddings == [0.5, 0.6, 0.7]
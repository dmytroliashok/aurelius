import pytest

from aurelius.common.llm.factory import create_llm
from aurelius.common.llm.openai_provider import OpenAIProvider


class TestCreateLLM:
    def test_returns_openai_provider(self):
        provider = create_llm(model="deepseek-chat")
        assert isinstance(provider, OpenAIProvider)
        assert provider.model_name == "deepseek-chat"

    def test_default_model(self):
        provider = create_llm()
        assert provider.model_name == "deepseek-chat"

    def test_custom_api_key(self):
        provider = create_llm(api_key="sk-test-key")
        assert isinstance(provider, OpenAIProvider)

    def test_custom_base_url(self):
        provider = create_llm(base_url="http://localhost:1234/v1")
        assert isinstance(provider, OpenAIProvider)
        assert provider._base_url == "http://localhost:1234/v1"

    def test_default_base_url(self):
        provider = create_llm()
        assert "deepseek" in provider._base_url

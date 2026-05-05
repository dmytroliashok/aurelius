"""Factory for creating LLM provider instances."""

from aurelius.common.llm.base import LLMProvider


def create_llm(model: str | None = None, api_key: str | None = None, base_url: str | None = None) -> LLMProvider:
    """Create an OpenAI-compatible LLM provider.

    All LLM calls use the OpenAI SDK protocol. Default endpoint is DeepSeek API.
    Override base_url for other OpenAI-compatible services (OpenAI, LM Studio, etc.).

    Args:
        model: Model name. Default: "deepseek-chat".
        api_key: API key. Uses env var if None.
        base_url: Base URL override. Default: DeepSeek API.

    Returns:
        An LLMProvider instance.
    """
    from aurelius.common.llm.openai_provider import OpenAIProvider

    return OpenAIProvider(model=model or "deepseek-chat", api_key=api_key, base_url=base_url)

"""OpenAI-compatible LLM provider (default: DeepSeek API)."""

import logging

from aurelius.common.llm.base import LLMProvider

logger = logging.getLogger(__name__)

# Default base URL for the OpenAI-compatible provider.
# All callers get DeepSeek unless they explicitly override.
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "deepseek-chat", api_key: str | None = None, base_url: str | None = None):
        self._model = model
        self._api_key = api_key
        self._base_url = base_url if base_url is not None else DEFAULT_BASE_URL
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            kwargs = {"base_url": self._base_url, "timeout": 120.0}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    @property
    def model_name(self) -> str:
        return self._model

    async def complete(self, prompt: str, *, system: str = "", max_tokens: int = 2000, temperature: float = 0.7) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return await self.complete_chat(messages, max_tokens=max_tokens, temperature=temperature)

    async def complete_chat(
        self, messages: list[dict], *, system: str = "", max_tokens: int = 2000, temperature: float = 0.7
    ) -> str:
        client = self._get_client()
        if system:
            messages = [{"role": "system", "content": system}] + messages

        response = await client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

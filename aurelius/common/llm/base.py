"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Unified interface for LLM completions."""

    @abstractmethod
    async def complete(self, prompt: str, *, system: str = "", max_tokens: int = 2000, temperature: float = 0.7) -> str:
        """Generate a completion from a single prompt.

        Args:
            prompt: The user prompt.
            system: Optional system prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text.
        """

    @abstractmethod
    async def complete_chat(
        self, messages: list[dict], *, system: str = "", max_tokens: int = 2000, temperature: float = 0.7
    ) -> str:
        """Generate a completion from a chat message list.

        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."} dicts.
            system: Optional system prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text.
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The model identifier."""

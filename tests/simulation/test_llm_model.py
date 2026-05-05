"""Unit test for the patched LLM model wrapper.

Concordia is only installed inside the simulation Docker image, not the
validator's runtime — so this test is skipped when ``gdm-concordia``
isn't importable. To run locally:

    pip install "gdm-concordia[openai]==2.4.0" && pytest tests/simulation/test_llm_model.py
"""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("concordia.contrib.language_models.openai.gpt_model")


def test_sample_text_overrides_reasoning_effort_to_max():
    """Whatever value upstream's ``sample_text`` / ``sample_choice``
    pass for ``reasoning_effort``, our subclass must rewrite it to
    ``max`` before calling the OpenAI client. This is the only behavior
    we are taking on; everything else must remain untouched."""
    from aurelius.simulation.llm_model import REASONING_EFFORT, AureliusGptLanguageModel

    assert REASONING_EFFORT == "max", (
        "REASONING_EFFORT changed away from 'max' — confirm DeepSeek still "
        "accepts the new value before merging this change."
    )

    # Build a model bypassing the upstream __init__ (which constructs an
    # OpenAI client and validates the API key). We only need a minimally
    # initialized instance for the override path.
    model = AureliusGptLanguageModel.__new__(AureliusGptLanguageModel)
    model._model_name = "deepseek-chat"
    model._client = MagicMock()
    model._verbosity = "low"
    model._measurements = None
    model._channel = "test"

    # Mock the OpenAI client's chat.completions.create so we can inspect
    # what reasoning_effort it actually received.
    fake_response = MagicMock()
    fake_response.choices = [MagicMock()]
    fake_response.choices[0].message.content = "ok"
    model._client.chat.completions.create.return_value = fake_response

    # Caller passes ``minimal`` (upstream's sample_text default) — our
    # override must rewrite it before the OpenAI call.
    result = model._sample_text(
        prompt="hello",
        reasoning_effort="minimal",
        verbosity="low",
    )
    assert result == "ok"
    call_kwargs = model._client.chat.completions.create.call_args.kwargs
    assert call_kwargs["reasoning_effort"] == "max", (
        f"reasoning_effort was not rewritten: got {call_kwargs['reasoning_effort']!r}"
    )

    # And the ``medium`` path that ``sample_choice`` uses.
    model._client.chat.completions.create.reset_mock()
    model._client.chat.completions.create.return_value = fake_response
    model._sample_text(
        prompt="hello",
        reasoning_effort="medium",
        verbosity="low",
    )
    call_kwargs = model._client.chat.completions.create.call_args.kwargs
    assert call_kwargs["reasoning_effort"] == "max"


def test_sample_text_handles_positional_reasoning_effort():
    """Defensive: if a future upstream switches to positional args,
    the override must still rewrite the value."""
    from aurelius.simulation.llm_model import REASONING_EFFORT, AureliusGptLanguageModel

    model = AureliusGptLanguageModel.__new__(AureliusGptLanguageModel)
    model._model_name = "deepseek-chat"
    model._client = MagicMock()
    model._verbosity = "low"
    model._measurements = None
    model._channel = "test"

    fake_response = MagicMock()
    fake_response.choices = [MagicMock()]
    fake_response.choices[0].message.content = "ok"
    model._client.chat.completions.create.return_value = fake_response

    # Positional: prompt, reasoning_effort, verbosity
    model._sample_text("hello", "minimal", "low")
    call_kwargs = model._client.chat.completions.create.call_args.kwargs
    assert call_kwargs["reasoning_effort"] == REASONING_EFFORT

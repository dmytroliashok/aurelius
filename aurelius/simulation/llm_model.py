"""LLM model wrapper used by the Concordia simulation entrypoint.

Concordia v2.4.0's ``GptLanguageModel.sample_text()`` hardcodes
``reasoning_effort='minimal'`` (and ``'medium'`` in ``sample_choice``).
DeepSeek's chat-completions API rejects ``'minimal'`` with HTTP 400 —
the only accepted variants are ``high``, ``low``, ``medium``, ``max``,
``xhigh``. Every simulation against DeepSeek failed on its first LLM
call, the entrypoint's exception handler wrote a 1-event fallback
transcript, and stage 7 (simulate) failed coherence on every miner.

This module rebinds ``reasoning_effort`` to a DeepSeek-acceptable
value before the upstream ``_sample_text`` runs, leaving the rest of
Concordia's behavior untouched.
"""

from __future__ import annotations

import os

from concordia.contrib.language_models.openai.gpt_model import GptLanguageModel
from concordia.utils import measurements as measurements_lib

# DeepSeek accepts: high, low, medium, max, xhigh. Picking ``max`` so the
# subnet's reasoning quality does not regress relative to upstream's
# ``minimal`` default — operator preference, set explicitly here so a
# future upstream bump cannot silently downgrade us.
REASONING_EFFORT = os.environ.get("LLM_REASONING_EFFORT", "max")


class AureliusGptLanguageModel(GptLanguageModel):
    """``GptLanguageModel`` with the ``reasoning_effort`` arg pinned.

    Both upstream call sites (``sample_text`` → ``minimal``,
    ``sample_choice`` → ``medium``) route through ``_sample_text``,
    so one override covers them. ``**kwargs`` keeps us robust to
    upstream signature drift on params we don't care about.
    """

    def _sample_text(self, *args, **kwargs) -> str:  # type: ignore[override]
        # ``reasoning_effort`` is the second positional arg in upstream's
        # signature (after ``prompt``). Today both callers pass it as a
        # kwarg, but cover the positional path too so a future upstream
        # refactor can't silently un-pin our override.
        if len(args) >= 2:
            args = (args[0], REASONING_EFFORT, *args[2:])
        else:
            kwargs["reasoning_effort"] = REASONING_EFFORT
        return super()._sample_text(*args, **kwargs)


def make_model(
    *,
    model_name: str,
    api_key: str | None,
    api_base: str | None,
    measurements: measurements_lib.Measurements | None = None,
) -> AureliusGptLanguageModel:
    """Construct the patched model. Centralized so the entrypoint and
    tests share the same factory."""
    return AureliusGptLanguageModel(
        model_name=model_name,
        api_key=api_key,
        api_base=api_base,
        measurements=measurements,
    )

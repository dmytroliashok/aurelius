"""Embedding service for scenario configs using sentence-transformers.

Produces deterministic, fixed-dimension embeddings regardless of agent count
by sorting agents alphabetically by name and mean-pooling.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingService:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            logger.info("Loaded embedding model: %s", self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        return self.model.encode(text, normalize_embeddings=True)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed multiple text strings."""
        return self.model.encode(texts, normalize_embeddings=True)

    def extract_field_embeddings(self, config: dict, parsed_config=None) -> dict[str, list[float]]:
        """Extract per-field embeddings for novelty checking.

        Uses the typed Pydantic model if available, falls back to dict access.
        Returns a dict mapping field names to embedding vectors.
        """
        field_embeddings: dict[str, list[float]] = {}
        if parsed_config is not None:
            field_embeddings["premise"] = self.embed_text(parsed_config.premise).tolist()
            for i, agent in enumerate(sorted(parsed_config.agents, key=lambda a: a.name)):
                field_embeddings[f"agent_goal_{i}"] = self.embed_text(agent.goal).tolist()
            for i, scene in enumerate(parsed_config.scenes):
                if scene.forced_choice:
                    fc = scene.forced_choice
                    fc_text = f"{fc.call_to_action} {' / '.join(fc.choices)}".strip()
                    field_embeddings[f"forced_choice_{i}"] = self.embed_text(fc_text).tolist()
        else:
            premise = config.get("premise", "")
            if premise:
                field_embeddings["premise"] = self.embed_text(premise).tolist()
            agents = sorted(config.get("agents", []), key=lambda a: a.get("name", ""))
            for i, agent in enumerate(agents):
                goal = agent.get("goal", "")
                if goal:
                    field_embeddings[f"agent_goal_{i}"] = self.embed_text(goal).tolist()
            for i, scene in enumerate(config.get("scenes", [])):
                fc = scene.get("forced_choice")
                if fc:
                    choices = fc.get("choices", [])
                    cta = fc.get("call_to_action", "")
                    fc_text = f"{cta} {' / '.join(choices)}".strip()
                    if fc_text:
                        field_embeddings[f"forced_choice_{i}"] = self.embed_text(fc_text).tolist()
        return field_embeddings

    def embed_config(self, config: dict) -> np.ndarray:
        """Embed a scenario config into a fixed-dimension vector.

        Strategy:
        1. Sort agents alphabetically by name for determinism
        2. Embed key fields: premise, tension, each agent's identity+goal
        3. Mean-pool all embeddings to a single fixed-dimension vector
        """
        texts = []

        # Premise
        premise = config.get("premise", "")
        if premise:
            texts.append(premise)

        # Tension context
        archetype = config.get("tension_archetype", "")
        description = config.get("tension_description", "")
        tension_text = f"{archetype}: {description}" if description else archetype
        if tension_text:
            texts.append(tension_text)

        # Agents sorted alphabetically by name
        agents = sorted(config.get("agents", []), key=lambda a: a.get("name", ""))
        for agent in agents:
            identity = agent.get("identity", "")
            goal = agent.get("goal", "")
            agent_text = f"{identity} {goal}".strip()
            if agent_text:
                texts.append(agent_text)

        # Forced choices
        for scene in config.get("scenes", []):
            fc = scene.get("forced_choice")
            if fc:
                choices = fc.get("choices", [])
                cta = fc.get("call_to_action", "")
                fc_text = f"{cta} {' / '.join(choices)}".strip()
                if fc_text:
                    texts.append(fc_text)

        if not texts:
            return np.zeros(self.dimension, dtype=np.float32)

        embeddings = self.embed_texts(texts)
        # Mean-pool to fixed dimension
        return np.mean(embeddings, axis=0).astype(np.float32)


# Module-level singleton (lazy-loaded)
_service: EmbeddingService | None = None


def get_embedding_service(model_name: str = DEFAULT_MODEL_NAME) -> EmbeddingService:
    global _service
    if _service is None or _service.model_name != model_name:
        _service = EmbeddingService(model_name)
    return _service

import numpy as np
import pytest

from aurelius.common.embeddings import EmbeddingService


def _valid_config(name="test_scenario") -> dict:
    return {
        "name": name,
        "tension_archetype": "justice_vs_mercy",
        "morebench_context": "Healthcare",
        "premise": "A doctor faces a difficult choice in a hospital with limited resources. " * 5,
        "agents": [
            {
                "name": "Dr. Chen",
                "identity": "I am a surgeon with 20 years of experience.",
                "goal": "I want to save the most lives possible.",
            },
            {
                "name": "Nurse Patel",
                "identity": "I am a senior nurse who advocates for patients.",
                "goal": "I want to ensure patient safety above all.",
            },
        ],
        "scenes": [{"steps": 3, "mode": "decision"}],
    }


@pytest.fixture(scope="module")
def service():
    return EmbeddingService()


class TestEmbeddingService:
    def test_embed_text_returns_vector(self, service):
        vec = service.embed_text("Hello world")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (service.dimension,)

    def test_embed_config_returns_fixed_dimension(self, service):
        vec = service.embed_config(_valid_config())
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (service.dimension,)

    def test_same_config_same_embedding(self, service):
        config = _valid_config()
        vec1 = service.embed_config(config)
        vec2 = service.embed_config(config)
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        assert similarity > 0.99  # Near-identical

    def test_different_configs_different_embeddings(self, service):
        config1 = _valid_config("scenario_alpha")
        config2 = _valid_config("scenario_beta")
        config2["premise"] = "An engineer must decide between safety and deadline pressure. " * 5
        config2["tension_archetype"] = "short_term_vs_long_term"

        vec1 = service.embed_config(config1)
        vec2 = service.embed_config(config2)
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        assert similarity < 0.95  # Distinct enough

    def test_agent_order_deterministic(self, service):
        """Agents sorted alphabetically — order in config shouldn't matter."""
        config1 = _valid_config()
        config2 = _valid_config()
        config2["agents"] = list(reversed(config2["agents"]))

        vec1 = service.embed_config(config1)
        vec2 = service.embed_config(config2)
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        assert similarity > 0.99  # Should be identical after sorting

    def test_empty_config(self, service):
        vec = service.embed_config({})
        assert vec.shape == (service.dimension,)
        assert np.allclose(vec, 0.0)

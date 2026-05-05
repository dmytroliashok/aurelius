"""Feature extraction for the quality classifier.

Produces a fixed-length feature vector from a scenario config, regardless
of agent or scene count. Three feature categories:

1. Structural: counts, lengths, presence flags
2. Semantic: embeddings of key text fields
3. Cross-field: pairwise similarities (agent-count-agnostic via summary stats)
"""

import numpy as np

from aurelius.common.enums import Philosophy, TensionArchetype

# Ordered enum values for one-hot encoding
_ARCHETYPE_VALUES = [a.value for a in TensionArchetype]
_PHILOSOPHY_VALUES = [p.value for p in Philosophy]

# Feature dimensions
N_STRUCTURAL = 15 + len(_ARCHETYPE_VALUES)  # counts + archetype one-hot
N_CROSS_FIELD = 12  # 3 groups × 4 stats (min/max/mean/std)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _summary_stats(values: list[float]) -> list[float]:
    """Return [min, max, mean, std] for a list of values. Returns [0,0,0,0] if empty."""
    if not values:
        return [0.0, 0.0, 0.0, 0.0]
    arr = np.array(values)
    return [float(arr.min()), float(arr.max()), float(arr.mean()), float(arr.std())]


def extract_structural_features(config: dict) -> np.ndarray:
    """Extract structural features from a config dict."""
    agents = config.get("agents", [])
    scenes = config.get("scenes", [])
    premise = config.get("premise", "")

    features = []

    # Counts
    features.append(len(agents))
    features.append(len(scenes))
    features.append(len(premise))
    features.append(sum(1 for s in scenes if s.get("forced_choice")))
    features.append(sum(s.get("steps", 0) for s in scenes))

    # Field presence
    features.append(1.0 if config.get("tension_description") else 0.0)
    features.append(1.0 if config.get("morebench_context") else 0.0)

    # Agent diversity
    philosophies = {a.get("philosophy", "") for a in agents}
    features.append(len(philosophies))
    features.append(1.0 if "" not in philosophies else 0.0)  # all agents have philosophy

    # Identity/goal lengths
    identity_lengths = [len(a.get("identity", "")) for a in agents]
    goal_lengths = [len(a.get("goal", "")) for a in agents]
    features.extend(_summary_stats(identity_lengths)[:2])  # min, max identity length
    features.extend(_summary_stats(goal_lengths)[:2])  # min, max goal length

    # Scene modes
    features.append(sum(1.0 for s in scenes if s.get("mode") == "reflection"))
    features.append(sum(1.0 for s in scenes if s.get("mode") == "decision"))

    # Tension archetype one-hot
    archetype = config.get("tension_archetype", "")
    for av in _ARCHETYPE_VALUES:
        features.append(1.0 if archetype == av else 0.0)

    return np.array(features, dtype=np.float32)


def extract_cross_field_features(config: dict, embedding_service=None) -> np.ndarray:
    """Extract cross-field similarity features using embeddings.

    Agent-count-agnostic: computes pairwise similarities and reduces to summary stats.
    """
    if embedding_service is None:
        return np.zeros(N_CROSS_FIELD, dtype=np.float32)

    premise = config.get("premise", "")
    agents = config.get("agents", [])
    scenes = config.get("scenes", [])

    # Embed texts
    premise_emb = embedding_service.embed_text(premise) if premise else np.zeros(embedding_service.dimension)
    goal_embs = []
    for agent in agents:
        goal = agent.get("goal", "")
        if goal:
            goal_embs.append(embedding_service.embed_text(goal))

    # Group 1: premise-to-goal similarities
    p2g_sims = [_cosine_similarity(premise_emb, g) for g in goal_embs]

    # Group 2: goal-to-goal similarities (pairwise)
    g2g_sims = []
    for i in range(len(goal_embs)):
        for j in range(i + 1, len(goal_embs)):
            g2g_sims.append(_cosine_similarity(goal_embs[i], goal_embs[j]))

    # Group 3: forced-choice option distance
    fc_sims = []
    for scene in scenes:
        fc = scene.get("forced_choice")
        if fc:
            choices = fc.get("choices", [])
            if len(choices) == 2:
                e1 = embedding_service.embed_text(choices[0])
                e2 = embedding_service.embed_text(choices[1])
                fc_sims.append(_cosine_similarity(e1, e2))

    features = []
    features.extend(_summary_stats(p2g_sims))
    features.extend(_summary_stats(g2g_sims))
    features.extend(_summary_stats(fc_sims))

    return np.array(features, dtype=np.float32)


def extract_features(config: dict, embedding_service=None) -> np.ndarray:
    """Extract the full feature vector for a scenario config.

    Args:
        config: Scenario config dict.
        embedding_service: Optional EmbeddingService for semantic features.
            If None, cross-field features are zeros.

    Returns:
        Fixed-length numpy array.
    """
    structural = extract_structural_features(config)
    cross_field = extract_cross_field_features(config, embedding_service)
    return np.concatenate([structural, cross_field])


def feature_dimension(embedding_service=None) -> int:
    """Return the total feature vector dimension."""
    return N_STRUCTURAL + N_CROSS_FIELD

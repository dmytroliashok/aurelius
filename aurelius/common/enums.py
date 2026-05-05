from enum import Enum


class TensionArchetype(str, Enum):
    """Eight named moral tension archetypes plus a custom option."""

    AUTONOMY_VS_BENEFICENCE = "autonomy_vs_beneficence"
    JUSTICE_VS_MERCY = "justice_vs_mercy"
    INDIVIDUAL_VS_COLLECTIVE = "individual_vs_collective"
    TRUTH_VS_LOYALTY = "truth_vs_loyalty"
    SHORT_TERM_VS_LONG_TERM = "short_term_vs_long_term"
    RIGHTS_VS_UTILITY = "rights_vs_utility"
    CARE_VS_FAIRNESS = "care_vs_fairness"
    LIBERTY_VS_EQUALITY = "liberty_vs_equality"
    CUSTOM = "custom"


class Philosophy(str, Enum):
    """Ten named moral philosophy frameworks plus empty (none)."""

    UTILITARIANISM = "utilitarianism"
    DEONTOLOGY = "deontology"
    VIRTUE_ETHICS = "virtue_ethics"
    CARE_ETHICS = "care_ethics"
    CONTRACTUALISM = "contractualism"
    NATURAL_LAW = "natural_law"
    PRAGMATISM = "pragmatism"
    EXISTENTIALISM = "existentialism"
    MORAL_RELATIVISM = "moral_relativism"
    DIVINE_COMMAND = "divine_command"
    NONE = ""


class SceneMode(str, Enum):
    """Scene execution modes."""

    DECISION = "decision"
    REFLECTION = "reflection"

"""Benchmark pipeline configuration."""

from dataclasses import dataclass, field


@dataclass
class FinetuneConfig:
    """Configuration for the SFT fine-tuning pipeline."""

    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    warmup_ratio: float = 0.03
    output_dir: str = "output/finetune"


@dataclass
class EvalConfig:
    """Configuration for MoReBench evaluation."""

    morebench_path: str = "data/morebench_public.json"
    num_scenarios: int = 500
    judge_model: str = "deepseek-chat"
    max_concurrent: int = 10


@dataclass
class BenchmarkPipelineConfig:
    """Full pipeline configuration."""

    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    batch_target_size: int = 50
    batch_min_size: int = 30
    influence_min_batch_size: int = 30

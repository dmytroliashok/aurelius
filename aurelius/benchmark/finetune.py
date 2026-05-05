"""SFT fine-tuning pipeline with LoRA.

Prepares datasets from simulation transcripts and fine-tunes a base LLM.
Heavy dependencies (transformers, peft) are imported lazily.
"""

import json
import logging
from pathlib import Path

from aurelius.benchmark.config import FinetuneConfig

logger = logging.getLogger(__name__)


def prepare_dataset(
    transcripts: list[dict],
    max_seq_length: int = 2048,
    min_rubric_score: float | None = None,
    rubric_scores: dict[int, float] | None = None,
) -> list[dict]:
    """Convert simulation transcripts into instruction-following format for SFT.

    Supports rejection sampling: if min_rubric_score is set, only transcripts
    whose index has a rubric score >= the threshold are included (Spec Section 3.5.2).

    Args:
        transcripts: List of transcript dicts (from Transcript.model_dump()).
        max_seq_length: Maximum token length hint (actual truncation handled by tokenizer).
        min_rubric_score: If set, exclude transcripts below this MoReBench rubric score.
        rubric_scores: Optional mapping of transcript index → rubric score for filtering.

    Returns:
        List of {"instruction": ..., "response": ...} dicts.
    """
    examples = []

    for idx, transcript in enumerate(transcripts):
        # Rejection sampling: skip transcripts below rubric threshold
        if min_rubric_score is not None and rubric_scores is not None:
            score = rubric_scores.get(idx, 0.0)
            if score < min_rubric_score:
                continue
        events = transcript.get("events", [])
        agent_names = transcript.get("agent_names", [])
        transcript.get("metadata", {})

        if not events:
            continue

        # Build the scenario context from narrations and scene starts
        context_parts = []
        for e in events:
            if e.get("type") in ("scene_start", "narration"):
                context_parts.append(e.get("content", ""))

        context = " ".join(context_parts).strip()

        # Extract agent actions as reasoning examples
        for agent_name in agent_names:
            agent_events = [e for e in events if e.get("agent") == agent_name]
            actions = [e.get("content", "") for e in agent_events if e.get("type") == "action" and e.get("content")]
            forced_choices = [e for e in agent_events if e.get("type") == "forced_choice"]

            if not actions:
                continue

            reasoning = "\n".join(f"- {a}" for a in actions)

            # Main example: scenario → agent's moral reasoning
            instruction = (
                f"You are {agent_name} in the following moral dilemma. "
                f"Explain your moral reasoning and the actions you would take.\n\n"
                f"Scenario: {context[:1000]}"
            )

            response_parts = [f"As {agent_name}, here is my moral reasoning:\n", reasoning]

            # Add forced choice if present
            for fc in forced_choices:
                fc_content = fc.get("content", "")
                fc_meta = fc.get("metadata", {})
                choices = fc_meta.get("choices", [])
                if fc_content and choices:
                    response_parts.append(
                        f"\nWhen faced with the choice between {' and '.join(choices)}, I decided: {fc_content}"
                    )

            examples.append(
                {
                    "instruction": instruction,
                    "response": "\n".join(response_parts),
                }
            )

    logger.info("Prepared %d training examples from %d transcripts", len(examples), len(transcripts))
    return examples


def save_dataset(examples: list[dict], output_path: str) -> None:
    """Save prepared dataset as JSONL."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    logger.info("Saved %d examples to %s", len(examples), output_path)


def finetune(dataset_path: str, config: FinetuneConfig | None = None) -> str:
    """Run SFT with LoRA on a prepared dataset.

    Args:
        dataset_path: Path to JSONL training data.
        config: Fine-tuning configuration. Uses defaults if None.

    Returns:
        Path to the output directory containing the fine-tuned adapter.
    """
    if config is None:
        config = FinetuneConfig()

    logger.info("Starting fine-tuning: base_model=%s, lora_rank=%d", config.base_model, config.lora_rank)

    # Lazy imports — these are heavy dependencies
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    # Load dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(example):
        text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
        return tokenizer(text, truncation=True, max_length=config.max_seq_length, padding="max_length")

    tokenized = dataset.map(tokenize, batched=False, remove_columns=dataset.column_names)

    # Model + LoRA
    model = AutoModelForCausalLM.from_pretrained(config.base_model, torch_dtype="auto", device_map="auto")

    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    logger.info("Fine-tuning complete. Adapter saved to %s", config.output_dir)
    return config.output_dir

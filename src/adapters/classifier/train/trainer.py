import logging
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import get_linear_schedule_with_warmup

from ..preparation import Example, ExampleDataset, encode
from ..preparation.trie import Encoded
from .batching import fitting_items, micro_batches, pick_token_budget
from .checkpoint import (
    BEST_NAME,
    CHECKPOINT_NAME,
    RUNS_DIRECTORY,
    RunState,
    adapter_directory,
    load_run_state,
    restore_optimizer_state,
    save_checkpoint,
)
from .evaluation import adapter_accuracy
from .losses import ReferenceTargets, classifier_loss, reference_targets
from .model import MODEL_DIRECTORY, load_model_and_tokenizer, pick_device
from .scorer import node_log_probs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainConfig:
    model_directory: Path = MODEL_DIRECTORY
    adapter_name: str = "classifier"
    lora_rank: int = 16
    lora_alpha: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    warmup_fraction: float = 0.10
    regularization_weight: float = 1.0
    ordinal_weight: float = 1.0
    batch_size: int = 32
    epochs: int = 3
    evaluation_interval_steps: int = 200
    validation_examples: int = 2000
    patience: int = 3
    seed: int = 20260922


def build_peft_model(base_model, config: TrainConfig, checkpoint_directory: Path) -> PeftModel:
    base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if checkpoint_directory.exists():
        return PeftModel.from_pretrained(base_model, checkpoint_directory, is_trainable=True)
    lora_config = LoraConfig(
        r=config.lora_rank, lora_alpha=config.lora_alpha, target_modules="all-linear", task_type="CAUSAL_LM"
    )
    return get_peft_model(base_model, lora_config)


def resumable_batches(
    dataset: ExampleDataset, batch_size: int, epochs: int, first_step: int
) -> Iterator[list[Example]]:
    steps_per_epoch = math.ceil(len(dataset) / batch_size)
    for epoch in range(first_step // steps_per_epoch, epochs):
        yield from dataset.batches(batch_size, epoch, max(first_step - epoch * steps_per_epoch, 0))


def cached_reference_targets(
    model, encoded_batch, references: dict[str, ReferenceTargets], batch
) -> list[ReferenceTargets]:
    missing = [
        (example, encoded) for example, encoded in zip(batch, encoded_batch) if example.identifier not in references
    ]
    if missing:
        with torch.no_grad(), model.disable_adapter():
            missing_log_probs = node_log_probs(model, [encoded for _, encoded in missing])
        for (example, encoded), log_probs in zip(missing, missing_log_probs):
            references[example.identifier] = reference_targets(log_probs, encoded)
    return [references[example.identifier] for example in batch]


def accumulate_micro_batch_gradients(
    model, micro_batch: list[tuple[Example, Encoded]], config: TrainConfig, references, batch_size: int
) -> float:
    examples = [example for example, _ in micro_batch]
    encoded_batch = [encoded for _, encoded in micro_batch]
    batch_references = cached_reference_targets(model, encoded_batch, references, examples)
    micro_batch_loss = (
        sum(
            classifier_loss(current, reference, encoded, example, config.regularization_weight, config.ordinal_weight)
            for current, reference, encoded, example in zip(
                node_log_probs(model, encoded_batch), batch_references, encoded_batch, examples
            )
        )
        / batch_size
    )
    micro_batch_loss.backward()
    return micro_batch_loss.item()


def accumulate_batch_gradients(
    model, tokenizer, batch: list[Example], config: TrainConfig, references, token_budget: float
) -> float:
    shuffled_batch = [example.shuffled() for example in batch]
    items = fitting_items([(example, encode(tokenizer, example)) for example in shuffled_batch], token_budget)
    return sum(
        accumulate_micro_batch_gradients(model, micro_batch, config, references, len(items))
        for micro_batch in micro_batches(items, token_budget)
    )


def evaluate_and_checkpoint(
    model, tokenizer, validation_dataset, token_budget, optimizer, scheduler, run_state: RunState, run_directory: Path
) -> None:
    model.eval()
    accuracy = adapter_accuracy(model, tokenizer, validation_dataset, token_budget)
    model.train()
    run_state.stale_evaluations += 1
    if accuracy > run_state.best_accuracy:
        run_state.best_accuracy = accuracy
        run_state.stale_evaluations = 0
        model.save_pretrained(run_directory / BEST_NAME)
    save_checkpoint(model, optimizer, scheduler, run_state, run_directory)
    logger.info("step %d validation accuracy=%.4f best=%.4f", run_state.step, accuracy, run_state.best_accuracy)


def train_adapter(config: TrainConfig, dataset: ExampleDataset, validation_dataset: ExampleDataset) -> Path:
    run_directory = RUNS_DIRECTORY / config.adapter_name
    checkpoint_directory = run_directory / CHECKPOINT_NAME
    torch.manual_seed(config.seed)
    base_model, tokenizer = load_model_and_tokenizer(config.model_directory)
    run_state = load_run_state(checkpoint_directory, config.batch_size)
    model = build_peft_model(base_model, config, checkpoint_directory).train()
    token_budget = pick_token_budget(
        model,
        tokenizer,
        dataset,
        lambda probe_batch: accumulate_batch_gradients(model, tokenizer, probe_batch, config, {}, math.inf),
    )
    model.zero_grad(set_to_none=True)

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=config.learning_rate, weight_decay=config.weight_decay, fused=pick_device() == "cuda"
    )
    total_steps = math.ceil(len(dataset) / run_state.batch_size) * config.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(config.warmup_fraction * total_steps), total_steps)
    restore_optimizer_state(optimizer, scheduler, checkpoint_directory)

    references = {}
    for batch in resumable_batches(dataset, run_state.batch_size, config.epochs, run_state.step):
        if run_state.stale_evaluations >= config.patience:
            break
        batch_loss = accumulate_batch_gradients(model, tokenizer, batch, config, references, token_budget)
        torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        run_state.step += 1
        logger.info("step %d/%d loss=%.4f", run_state.step, total_steps, batch_loss)
        if run_state.step % config.evaluation_interval_steps == 0 or run_state.step == total_steps:
            evaluate_and_checkpoint(
                model, tokenizer, validation_dataset, token_budget, optimizer, scheduler, run_state, run_directory
            )

    return adapter_directory(config.adapter_name)

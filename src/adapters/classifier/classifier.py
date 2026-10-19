from pathlib import Path

from .preparation import ExampleDataset
from .train import MODEL_DIRECTORY, TrainConfig, train_adapter
from .validation import validate_adapters

DEFAULT_SEED = 20260922


def preprocess(split: str = "train", max_examples: int | None = None, seed: int = DEFAULT_SEED) -> ExampleDataset:
    return ExampleDataset(split, max_examples, seed)


def train(max_examples: int | None = None, config: TrainConfig = TrainConfig()) -> Path:
    return train_adapter(
        config,
        preprocess("train", max_examples, config.seed),
        preprocess("validation", config.validation_examples, config.seed),
    )


def validate(
    split: str = "test",
    max_examples: int | None = None,
    adapter_names: tuple[str, ...] = ("classifier",),
    model_directory: Path = MODEL_DIRECTORY,
    seed: int = DEFAULT_SEED,
) -> dict[str, float]:
    return validate_adapters(model_directory, adapter_names, preprocess(split, max_examples, seed))

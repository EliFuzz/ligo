from pathlib import Path

from ..preparation import ExampleDataset, encode
from ..train.batching import pick_token_budget
from ..train.evaluation import adapter_accuracy, predicted_options
from ..train.model import load_adapters, load_model_and_tokenizer


def validate_adapters(
    model_directory: Path, adapter_names: tuple[str, ...], dataset: ExampleDataset
) -> dict[str, float]:
    model, tokenizer = load_model_and_tokenizer(model_directory)
    peft_model = load_adapters(model, adapter_names)
    token_budget = pick_token_budget(
        peft_model,
        tokenizer,
        dataset,
        lambda probe_batch: predicted_options(peft_model, [encode(tokenizer, example) for example in probe_batch]),
    )
    accuracies = {}
    for adapter_name in adapter_names:
        peft_model.set_adapter(adapter_name)
        accuracies[adapter_name] = adapter_accuracy(peft_model, tokenizer, dataset, token_budget)
    return accuracies

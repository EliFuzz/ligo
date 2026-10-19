import torch

from ..preparation import ExampleDataset, encode
from ..preparation.trie import Encoded
from .batching import fitting_items, micro_batches
from .scorer import node_log_probs, tree_log_q

EVALUATION_READ_SIZE = 64


@torch.no_grad()
def predicted_options(model, encoded_batch: list[Encoded]) -> list[int]:
    return [
        tree_log_q(log_probs, encoded).argmax().item()
        for encoded, log_probs in zip(encoded_batch, node_log_probs(model, encoded_batch))
    ]


def adapter_accuracy(model, tokenizer, dataset: ExampleDataset, token_budget: int) -> float:
    correct = 0
    evaluated = 0
    for batch in dataset.batches(EVALUATION_READ_SIZE):
        items = fitting_items([(example, encode(tokenizer, example)) for example in batch], token_budget)
        evaluated += len(items)
        for micro_batch in micro_batches(items, token_budget):
            predictions = predicted_options(model, [encoded for _, encoded in micro_batch])
            correct += sum(prediction == example.gold for prediction, (example, _) in zip(predictions, micro_batch))
    return correct / evaluated

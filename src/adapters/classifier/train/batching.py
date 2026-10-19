import gc
import os
from collections.abc import Callable, Iterator
from dataclasses import replace

import torch

from ..preparation import Example, ExampleDataset, encode
from ..preparation.trie import Encoded

MEMORY_FRACTION = 0.5
PROBE_SAMPLE_SIZE = 64
PROBE_TOKEN_SIZES = [512 * 2**power for power in range(8)]
MPS_PROBE_TOKEN_SIZES = [512 * 2**power for power in range(4)]


def sequence_length(encoded: Encoded) -> int:
    return len(encoded.prompt_ids) + max(len(suffix) for suffix in encoded.suffixes)


def example_cost(encoded: Encoded) -> int:
    return len(encoded.suffixes) * sequence_length(encoded)


def fitting_items(items: list[tuple[Example, Encoded]], token_budget: int) -> list[tuple[Example, Encoded]]:
    return [item for item in items if example_cost(item[1]) <= token_budget]


def limit_device_memory(device_type: str) -> None:
    if device_type == "cuda":
        torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION)
    if device_type == "mps":
        physical_memory_bytes = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        torch.mps.set_per_process_memory_fraction(
            MEMORY_FRACTION * physical_memory_bytes / torch.mps.recommended_max_memory()
        )


def release_device_memory(device_type: str) -> None:
    gc.collect()
    if device_type == "cuda":
        torch.cuda.empty_cache()
    if device_type == "mps":
        torch.mps.empty_cache()


def fits_in_memory(run_probe: Callable[[list[Example]], object], probe_example: Example, device_type: str) -> bool:
    try:
        run_probe([probe_example])
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        return False
    finally:
        release_device_memory(device_type)
    return True


def pick_token_budget(model, tokenizer, dataset: ExampleDataset, run_probe: Callable[[list[Example]], object]) -> int:
    device_type = next(model.parameters()).device.type
    limit_device_memory(device_type)
    probe_token_sizes = MPS_PROBE_TOKEN_SIZES if device_type == "mps" else PROBE_TOKEN_SIZES
    seed_example = min(
        next(dataset.batches(PROBE_SAMPLE_SIZE)), key=lambda example: example_cost(encode(tokenizer, example))
    )
    seed_cost = example_cost(encode(tokenizer, seed_example))
    token_budget = 0
    for probe_tokens in probe_token_sizes:
        probe_example = replace(seed_example, state=seed_example.state * -(-probe_tokens // seed_cost))
        if not fits_in_memory(run_probe, probe_example, device_type):
            break
        token_budget = example_cost(encode(tokenizer, probe_example))
    if not token_budget:
        raise MemoryError("Not enough device memory to process the smallest probe")
    return token_budget


def micro_batches(items: list[tuple[Example, Encoded]], token_budget: int) -> Iterator[list[tuple[Example, Encoded]]]:
    group = []
    sequence_count = 0
    for item in sorted(items, key=lambda item: sequence_length(item[1])):
        item_sequence_count = len(item[1].suffixes)
        if group and (sequence_count + item_sequence_count) * sequence_length(item[1]) > token_budget:
            yield group
            group = []
            sequence_count = 0
        group.append(item)
        sequence_count += item_sequence_count
    if group:
        yield group

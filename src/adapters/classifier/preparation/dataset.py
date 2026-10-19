import json
import random
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .example import Example

DATASET_DIRECTORY = Path(__file__).parents[1] / "dataset"
HOLDOUT_INTERVAL = 20


def row_to_example(row: dict) -> Example:
    state = {name: value for name, value in row["state"].items() if value is not None}
    return Example(
        identifier=row["id"],
        state=json.dumps(state, ensure_ascii=False),
        instruction=row["question"],
        options=row["options"],
        target=row["target"],
        kind=row["kind"],
    )


def split_rows(dataset_directory: Path, split: str) -> tuple[pa.Table, range | list[int]] | None:
    split_paths = sorted(dataset_directory.glob(f"{split}-*.parquet"))
    has_validation = any(dataset_directory.glob("validation-*.parquet"))
    if split_paths and (split != "train" or has_validation):
        table = pq.read_table(split_paths, memory_map=True)
        return table, range(table.num_rows)
    if has_validation or split not in ("train", "validation"):
        return None
    table = pq.read_table(sorted(dataset_directory.glob("train-*.parquet")), memory_map=True)
    held_out_rows = range(0, table.num_rows, HOLDOUT_INTERVAL)
    if split == "validation":
        return table, held_out_rows
    return table, sorted(set(range(table.num_rows)) - set(held_out_rows))


class ExampleDataset:
    def __init__(self, split: str, max_examples: int | None, seed: int):
        selections = [
            selection
            for dataset_directory in sorted(DATASET_DIRECTORY.iterdir())
            if dataset_directory.is_dir() and (selection := split_rows(dataset_directory, split))
        ]
        self.tables = [table for table, _ in selections]
        row_locations = [
            (table_index, row_index)
            for table_index, (_, row_indices) in enumerate(selections)
            for row_index in row_indices
        ]
        example_count = min(max_examples or len(row_locations), len(row_locations))
        self.seed = seed
        self.row_locations = random.Random(seed).sample(row_locations, example_count)

    def __len__(self) -> int:
        return len(self.row_locations)

    def batches(self, batch_size: int, epoch: int = 0, first_batch: int = 0) -> Iterator[list[Example]]:
        epoch_locations = random.Random(self.seed + epoch).sample(self.row_locations, len(self))
        for start in range(first_batch * batch_size, len(self), batch_size):
            yield [
                row_to_example(self.tables[table_index].slice(row_index, 1).to_pylist()[0])
                for table_index, row_index in epoch_locations[start : start + batch_size]
            ]

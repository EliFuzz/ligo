import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

RUNS_DIRECTORY = Path(__file__).parents[1] / "runs"
CHECKPOINT_NAME = "checkpoint"
BEST_NAME = "best"
RUN_STATE_FILE = "run_state.json"
OPTIMIZER_STATE_FILE = "optimizer_state.pt"


def adapter_directory(adapter_name: str) -> Path:
    return RUNS_DIRECTORY / adapter_name / BEST_NAME


@dataclass
class RunState:
    batch_size: int
    step: int = 0
    best_accuracy: float = -1.0
    stale_evaluations: int = 0


def load_run_state(checkpoint_directory: Path, batch_size: int) -> RunState:
    state_path = checkpoint_directory / RUN_STATE_FILE
    if not state_path.exists():
        return RunState(batch_size=batch_size)
    return RunState(**json.loads(state_path.read_text()))


def restore_optimizer_state(optimizer, scheduler, checkpoint_directory: Path) -> None:
    state_path = checkpoint_directory / OPTIMIZER_STATE_FILE
    if not state_path.exists():
        return
    saved_state = torch.load(state_path)
    optimizer.load_state_dict(saved_state["optimizer"])
    scheduler.load_state_dict(saved_state["scheduler"])


def save_checkpoint(model, optimizer, scheduler, run_state: RunState, run_directory: Path) -> None:
    pending_directory = run_directory / f"{CHECKPOINT_NAME}.pending"
    model.save_pretrained(pending_directory)
    torch.save(
        {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        pending_directory / OPTIMIZER_STATE_FILE,
    )
    (pending_directory / RUN_STATE_FILE).write_text(json.dumps(asdict(run_state)))
    shutil.rmtree(run_directory / CHECKPOINT_NAME, ignore_errors=True)
    pending_directory.replace(run_directory / CHECKPOINT_NAME)

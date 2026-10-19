from pathlib import Path
import platform

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .checkpoint import adapter_directory

MODEL_DIRECTORY = Path(__file__).parents[4] / "models" / "qwen"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        raise RuntimeError("Apple Silicon GPU unavailable; refusing to run training on slow CPU")
    return "cpu"


def load_model_and_tokenizer(model_directory):
    torch.set_float32_matmul_precision("high")
    device = pick_device()
    dtype = torch.float32 if device == "cpu" else torch.float16 if device == "mps" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    model = AutoModelForCausalLM.from_pretrained(model_directory, dtype=dtype)
    return model.to(device).eval(), tokenizer


def load_adapters(model, adapter_names: tuple[str, ...]) -> PeftModel:
    peft_model = PeftModel.from_pretrained(model, adapter_directory(adapter_names[0]), adapter_name=adapter_names[0])
    for adapter_name in adapter_names[1:]:
        peft_model.load_adapter(adapter_directory(adapter_name), adapter_name=adapter_name)
    return peft_model.eval()

"""
Singleton model loader for the DTS Scoring Service.

The model (LLaMA-3.1-8B-Instruct in NF4 4-bit) is loaded once at application
startup and reused for every request.  GPU / CPU is detected automatically from
the MODEL_DEVICE environment variable (defaults to "cuda" if CUDA is available,
otherwise "cpu").
"""
import logging
import os
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from dtr_analyzer import DeepThinkingAnalyzer, DTSConfig

logger = logging.getLogger(__name__)

# ── Singleton state ───────────────────────────────────────────────────────────

_analyzer: Optional[DeepThinkingAnalyzer] = None
_model_name: str = ""
_device: str = ""
_num_layers: int = 0


def _resolve_device() -> str:
    """Return 'cuda' if a GPU is available, else 'cpu'."""
    env_device = os.getenv("MODEL_DEVICE", "auto")
    if env_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return env_device


def load_model() -> None:
    """
    Load LLaMA-3.1-8B-Instruct (NF4 4-bit) and initialise the DeepThinkingAnalyzer.

    Override the default model by setting MODEL_NAME_OR_PATH in the environment.
    Override the device with MODEL_DEVICE (auto | cuda | cpu).
    """
    global _analyzer, _model_name, _device, _num_layers

    _model_name = os.getenv(
        "MODEL_NAME_OR_PATH",
        "meta-llama/Llama-3.1-8B-Instruct",
    )
    _device = _resolve_device()

    logger.info("Loading model %s on %s …", _model_name, _device)

    tokenizer = AutoTokenizer.from_pretrained(_model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Quantisation config (NF4 4-bit) ──────────────────────────────────────
    if _device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            _model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    else:
        # CPU fallback — full precision (slow; for development / testing only)
        model = AutoModelForCausalLM.from_pretrained(
            _model_name,
            device_map="cpu",
            torch_dtype=torch.float32,
        )

    model.eval()

    dts_config = DTSConfig(
        settling_threshold=float(os.getenv("DTS_SETTLING_THRESHOLD", "0.5")),
        depth_fraction=float(os.getenv("DTS_DEPTH_FRACTION", "0.85")),
        device=_device,
    )

    _analyzer = DeepThinkingAnalyzer(
        model=model,
        tokenizer=tokenizer,
        config=dts_config,
        device=_device,
    )

    _num_layers = _analyzer.num_layers
    logger.info(
        "Model loaded — %d transformer layers, device=%s", _num_layers, _device
    )


def get_analyzer() -> DeepThinkingAnalyzer:
    """Return the global DeepThinkingAnalyzer; raises RuntimeError if not loaded."""
    if _analyzer is None:
        raise RuntimeError(
            "Model not loaded. Ensure load_model() was called at startup."
        )
    return _analyzer


def get_model_info() -> Tuple[str, str, int]:
    """Return (model_name, device, num_layers)."""
    return _model_name, _device, _num_layers

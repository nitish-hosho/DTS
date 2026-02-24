"""
DTS Reasoning Summariser
========================
Uses the already-loaded LLaMA model to interpret the raw DTS metrics
into a short, actionable natural-language analysis.

The meta-prompt is kept deliberately small (well under 200 tokens) so the
generation is fast and does not compete for VRAM with the scoring pass.
"""
import gc
import logging
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from schemas import ScoreResponse

logger = logging.getLogger("dts_service.reasoner")

# ── System instruction (fixed, ~90 tokens) ───────────────────────────────────
_SYSTEM = (
    "You are a concise AI analyst. Given Deep Thinking Score (DTS) metrics for "
    "a language-model response, write 2-3 sentences that: "
    "(1) interpret what the score means for this specific prompt, "
    "(2) identify which token types drove deep-layer settling, and "
    "(3) give one concrete, actionable recommendation for the prompt author. "
    "Be specific. Do not repeat the numbers verbatim."
)


def _build_meta_prompt(result: "ScoreResponse", prompt_preview: str) -> str:
    """Build a compact summary prompt from the scoring result."""

    # Top-5 deep tokens by settling layer (deepest first)
    top_deep = sorted(result.deep_tokens, key=lambda t: -t.settling_layer)[:5]
    deep_examples = ", ".join(
        f"'{t.token.strip()}' (layer {t.settling_layer})" for t in top_deep
    )

    # Shallow token count
    shallow_count = result.total_tokens - result.num_deep_tokens

    meta = (
        f"Prompt (excerpt): \"{prompt_preview[:120].strip()}\"\n\n"
        f"DTS Score: {result.dts_score} ({result.category})\n"
        f"Total response tokens: {result.total_tokens} "
        f"| Deep: {result.num_deep_tokens} | Shallow: {shallow_count}\n"
        f"Avg settling depth: {result.avg_settling_depth:.1f}/{result.num_layers} layers "
        f"(SDF={result.settling_depth_fraction:.3f})\n"
        f"Deepest tokens: {deep_examples}\n\n"
        f"Provide your 2-3 sentence actionable analysis:"
    )
    return meta


def generate_reasoning(
    result: "ScoreResponse",
    prompt_preview: str,
    max_new_tokens: int = 120,
) -> str:
    """
    Generate an actionable reasoning summary for the DTS result.

    Uses the singleton DeepThinkingAnalyzer's model + tokenizer directly —
    no extra VRAM cost beyond the already-loaded weights.

    Returns the generated summary string, or a fallback message on error.
    """
    from model_loader import get_analyzer  # lazy import to avoid circular

    try:
        analyzer = get_analyzer()
        model = analyzer.model
        tokenizer = analyzer.tokenizer

        meta_prompt = _build_meta_prompt(result, prompt_preview)

        # Apply the chat template so LLaMA responds as an assistant
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": meta_prompt},
        ]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(formatted, return_tensors="pt")
        try:
            primary_device = next(model.parameters()).device
        except StopIteration:
            primary_device = torch.device("cpu")

        input_ids = inputs["input_ids"].to(primary_device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # greedy — deterministic & fast
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        new_tokens = output_ids[0, input_ids.shape[1]:]
        summary = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        logger.info("Reasoning summary generated (%d tokens)", len(new_tokens))
        return summary

    except Exception as exc:
        logger.warning("Reasoning generation failed: %s", exc)
        return (
            f"DTS={result.dts_score} ({result.category}): "
            f"{result.num_deep_tokens}/{result.total_tokens} tokens required "
            f"deep-layer settling (avg depth {result.avg_settling_depth:.1f}/{result.num_layers})."
        )
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

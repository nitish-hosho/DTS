#!/usr/bin/env python3
"""
DTS Complexity Analysis on Golden Dataset (Turn-wise, RAG-aware)
================================================================

Measures Deep Thinking Score (DTS) as a proxy for response complexity
for each conversation turn in the golden dataset.

For every (User query → Bot response) turn the script:
  1. Reconstructs the full RAG-augmented prompt:
       [SYSTEM HEADER] + [KB CHUNKS] + [CONVERSATION HISTORY] + "User: <query>\nBot:"
  2. Uses *teacher-forcing* (forced-decode) to evaluate DTS over the
     known bot-response tokens — no sampling needed.
  3. Writes one CSV row per turn with complexity metrics.

RAG Modifications vs. vanilla DTS
----------------------------------
• KB retrieval is performed per-turn using the user query (+ prior bot context)
  so the prompt contains the same grounding the original bot had.
• `DeepThinkingAnalyzer.analyze_forced_decode()` replaces `analyze_prompt()`:
  it measures how many response tokens required deep-layer settling when the
  model is conditioned on the KB-enriched context, not just the user query.
• An extra column `rag_complexity_delta` estimates how much harder the
  KB-conditioned prompt is vs. a KB-free prompt (optional, requires --compare_no_rag).

Usage
-----
    python dts_complexity_golden_dataset.py \\
        --model_name_or_path "meta-llama/Llama-2-7b-chat-hf" \\
        --excel_file Golden_dataset_100.xlsx \\
        --output_csv dts_complexity_results.csv \\
        --kb_path knowledge_base/airline_faqs_markdown/ \\
        --max_rows 20 \\
        --max_new_tokens_per_turn 80 \\
        --device cuda

    # Dry-run without a real LLM (heuristic complexity):
    python dts_complexity_golden_dataset.py \\
        --excel_file Golden_dataset_100.xlsx \\
        --output_csv dts_complexity_results.csv \\
        --dry_run
"""

import argparse
import json
import logging
import os
import pickle
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# ── DTS core ──────────────────────────────────────────────────────────────────
from dtr_analyzer import (
    DeepThinkingAnalyzer,
    DTSConfig,
    PromptDTSAnalysis,
    DTSComparator,
    KBAttentionResult,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("dts_complexity")


# ─────────────────────────────────────────────────────────────────────────────
# Constants / defaults
# ─────────────────────────────────────────────────────────────────────────────

KB_DEFAULT_PATH = "/home/nitish/work/hosho/knowledge_base/airline_faqs_markdown/"
KB_EMBED_CACHE  = "/home/nitish/work/hosho/dts_kb_embed_cache.pkl"
EMBED_MODEL_ID  = "sentence-transformers/all-MiniLM-L6-v2"

# RAG prompt template (system header kept brief so it doesn't dominate context)
RAG_PROMPT_TEMPLATE = """\
[SYSTEM]
You are an airline customer-support assistant. Answer based ONLY on the knowledge base excerpts below.

[KNOWLEDGE BASE]
{kb_context}

[CONVERSATION]
{conversation_history}User: {user_query}
Bot:"""

# Prompt template when no KB chunks are available
NO_KB_PROMPT_TEMPLATE = """\
[SYSTEM]
You are an airline customer-support assistant.

[CONVERSATION]
{conversation_history}User: {user_query}
Bot:"""


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConversationTurn:
    """A single human→bot turn extracted from a transcript."""
    turn_idx: int               # 1-based
    user_query: str
    bot_response: str
    history_before: str         # All prior turns as a formatted string


@dataclass
class TurnComplexityResult:
    """DTS complexity metrics for a single conversation turn."""
    conversation_id: Any
    turn_idx: int
    user_query_preview: str
    bot_response_preview: str
    bot_response_length_chars: int
    bot_response_length_tokens: int
    num_kb_chunks_retrieved: int
    kb_top_similarity: float
    kb_chunks_preview: str          # First chunk, truncated
    # ── DTS metrics ──────────────────────────
    dts: float                      # Deep Thinking Score  (0–1)
    avg_settling_depth: float       # Mean settling layer index
    settling_depth_fraction: float  # Relative to total layers (0–1)
    num_deep_thinking_tokens: int   # Count of deep-thinking response tokens
    total_tokens_analyzed: int
    complexity_category: str        # shallow / moderate / deep
    # ── Dataset labels ───────────────────────
    ground_truth_accuracy: int      # 1 = bot correct, 0 = bot incorrect
    ground_truth_label: str         # PASS / FAIL
    # ── Meta ─────────────────────────────────
    analysis_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error: str = ""
    # ── KB Attention Mass (KAM) — requires --attn_impl=sdpa ──────────────────
    # KAM = fraction of response attention that lands on KB-chunk tokens.
    # High DTS + Low KAM = deep reasoning but KB ignored => strongest FAIL signal.
    kb_attention_mass: float = 0.0            # mean KAM over all layers/heads
    kb_attention_mass_early: float = 0.0      # layers 1–floor(L/3): surface matching
    kb_attention_mass_mid: float = 0.0        # layers floor(L/3)–floor(2L/3): semantic retrieval
    kb_attention_mass_late: float = 0.0       # layers floor(2L/3)–L: copying/reasoning
    kb_attention_mass_no_bos: float = 0.0     # BOS-excluded KAM (corrects attention sink)
    kb_attention_entropy: float = 0.0         # mean attention entropy (high=scanning, low=focused)
    dts_x_inv_kam: float = 0.0                # DTS * (1 - KAM): high = deep but KB-ignoring => FAIL


# ─────────────────────────────────────────────────────────────────────────────
# Transcript parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

# Regex that matches speaker prefixes robustly
_USER_RE  = re.compile(r"^(User|Customer|Human)\s*:", re.IGNORECASE)
_BOT_RE   = re.compile(r"^(Bot|Agent|Assistant|Representative)\s*:", re.IGNORECASE)


def parse_transcript_turns(transcript: str) -> List[ConversationTurn]:
    """
    Parse a raw transcript string into ordered (user_query, bot_response) pairs.

    Supports multi-line utterances — a new speaker tag terminates the previous
    speaker's accumulation buffer.

    Returns list of ConversationTurn objects, one per user→bot exchange.
    """
    if not transcript or not transcript.strip():
        return []

    lines = transcript.split("\n")
    # Each element: {"speaker": "user"|"bot", "text": str}
    utterances: List[Dict[str, str]] = []
    current_speaker: Optional[str] = None
    current_text: List[str] = []

    def flush():
        if current_speaker and current_text:
            utterances.append({
                "speaker": current_speaker,
                "text": " ".join(current_text).strip(),
            })

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        if _USER_RE.match(line_stripped):
            flush()
            current_speaker = "user"
            remainder = re.split(r"\s*:\s*", line_stripped, maxsplit=1)
            current_text = [remainder[1].strip()] if len(remainder) > 1 else []
        elif _BOT_RE.match(line_stripped):
            flush()
            current_speaker = "bot"
            remainder = re.split(r"\s*:\s*", line_stripped, maxsplit=1)
            current_text = [remainder[1].strip()] if len(remainder) > 1 else []
        else:
            # Continuation of previous speaker
            if current_speaker:
                current_text.append(line_stripped)

    flush()  # commit final utterance

    # Group into user→bot pairs
    turns: List[ConversationTurn] = []
    history_parts: List[str] = []
    i = 0

    while i < len(utterances):
        utt = utterances[i]

        if utt["speaker"] == "user":
            user_query = utt["text"]
            bot_response = ""

            # Look ahead for bot response
            if i + 1 < len(utterances) and utterances[i + 1]["speaker"] == "bot":
                bot_response = utterances[i + 1]["text"]
                i += 2
            else:
                i += 1  # lone user utterance, no bot reply found

            if not user_query:
                continue

            turns.append(ConversationTurn(
                turn_idx=len(turns) + 1,
                user_query=user_query,
                bot_response=bot_response,
                history_before="".join(history_parts),
            ))

            # Append this exchange to history for subsequent turns
            history_parts.append(f"User: {user_query}\n")
            if bot_response:
                history_parts.append(f"Bot: {bot_response}\n")
        else:
            # Bot utterance without a preceding user turn — skip
            i += 1

    return turns


# ─────────────────────────────────────────────────────────────────────────────
# KB retrieval (local Markdown files)
# ─────────────────────────────────────────────────────────────────────────────

class LocalKBRetriever:
    """
    Lightweight KB retriever backed by sentence-transformers cosine similarity.

    Loads all Markdown files under `kb_path`, splits them into paragraphs,
    embeds them once (or loads from cache), then answers per-turn queries.
    """

    def __init__(
        self,
        kb_path: str,
        embed_model_id: str = EMBED_MODEL_ID,
        cache_path: str = KB_EMBED_CACHE,
        device: str = "cpu",
    ):
        self.kb_path   = Path(kb_path)
        self.device    = device
        self.cache_path = cache_path

        logger.info(f"Loading KB retrieval model: {embed_model_id}")
        from sentence_transformers import SentenceTransformer
        self.embed_model = SentenceTransformer(embed_model_id, device=device)

        self.chunks: List[str] = []
        self.embeddings: Optional[torch.Tensor] = None
        self._load_kb()

    # ── KB loading ────────────────────────────────────────────────────────────

    def _load_kb(self):
        """Load KB chunks and their embeddings (from cache if available)."""
        md_files = list(self.kb_path.glob("**/*.md")) + list(self.kb_path.glob("**/*.txt"))
        if not md_files:
            logger.warning(f"No Markdown/text files found under {self.kb_path}")
            return

        # Collect chunks
        for fp in sorted(md_files):
            try:
                text = fp.read_text(encoding="utf-8", errors="ignore")
                # Split on blank lines → paragraphs
                paragraphs = re.split(r"\n{2,}", text)
                for para in paragraphs:
                    para = para.strip()
                    # Skip headers-only lines and very short fragments
                    if len(para) >= 60 and not re.match(r"^#{1,4}\s", para):
                        self.chunks.append(para)
            except Exception as e:
                logger.debug(f"Could not read {fp}: {e}")

        logger.info(f"Loaded {len(self.chunks)} KB chunks from {len(md_files)} files")

        if not self.chunks:
            return

        # Try cache
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    cached = pickle.load(f)
                if cached.get("num_chunks") == len(self.chunks):
                    self.embeddings = cached["embeddings"]
                    logger.info("Loaded KB embeddings from cache")
                    return
            except Exception as e:
                logger.warning(f"Cache load failed ({e}), recomputing…")

        # Compute embeddings
        logger.info("Computing KB embeddings (this may take a while)…")
        embs = self.embed_model.encode(
            self.chunks,
            batch_size=64,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        self.embeddings = embs.cpu()

        try:
            with open(self.cache_path, "wb") as f:
                pickle.dump({"num_chunks": len(self.chunks), "embeddings": self.embeddings}, f)
            logger.info(f"Cached KB embeddings to {self.cache_path}")
        except Exception as e:
            logger.warning(f"Could not save cache: {e}")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_sim: float = 0.25,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top-k KB chunks relevant to `query`.

        Returns list of (chunk_text, similarity_score) sorted descending.
        """
        if not self.chunks or self.embeddings is None:
            return []

        q_emb = self.embed_model.encode(
            [query],
            convert_to_tensor=True,
            normalize_embeddings=True,
        ).cpu()

        sims = torch.mm(q_emb, self.embeddings.T)[0]  # [N]
        top_indices = sims.argsort(descending=True)[:top_k]

        results = []
        for idx in top_indices:
            sim = sims[idx].item()
            if sim >= min_sim:
                results.append((self.chunks[idx], sim))

        return results


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def build_rag_prompt(
    user_query: str,
    conversation_history: str,
    kb_chunks: List[Tuple[str, float]],
    max_kb_chars: int = 3000,
) -> str:
    """
    Assemble the RAG-augmented prompt fed to the model.

    KB chunks are concatenated up to `max_kb_chars` characters so the full
    context window is not dominated by retrieved text on long conversations.
    """
    if kb_chunks:
        kb_text_parts = []
        total = 0
        for chunk_text, sim in kb_chunks:
            if total + len(chunk_text) > max_kb_chars:
                remaining = max_kb_chars - total
                if remaining > 100:
                    kb_text_parts.append(chunk_text[:remaining] + "…")
                break
            kb_text_parts.append(chunk_text)
            total += len(chunk_text)

        kb_context = "\n\n---\n\n".join(kb_text_parts)
        return RAG_PROMPT_TEMPLATE.format(
            kb_context=kb_context,
            conversation_history=conversation_history,
            user_query=user_query,
        )
    else:
        return NO_KB_PROMPT_TEMPLATE.format(
            conversation_history=conversation_history,
            user_query=user_query,
        )


def build_rag_prompt_with_offsets(
    user_query: str,
    conversation_history: str,
    kb_chunks: List[Tuple[str, float]],
    max_kb_chars: int = 3000,
) -> Tuple[str, int, int]:
    """
    Same as build_rag_prompt() but additionally returns the character span
    [kb_char_start, kb_char_end) of the KB content section within the assembled
    prompt string.  Used by analyze_kb_attention() to locate KB token indices.

    Returns:
        (prompt_str, kb_char_start, kb_char_end)
        If no KB chunks available, returns (prompt_str, 0, 0).
    """
    if not kb_chunks:
        prompt = NO_KB_PROMPT_TEMPLATE.format(
            conversation_history=conversation_history,
            user_query=user_query,
        )
        return prompt, 0, 0

    kb_text_parts: List[str] = []
    total = 0
    for chunk_text, sim in kb_chunks:
        if total + len(chunk_text) > max_kb_chars:
            remaining = max_kb_chars - total
            if remaining > 100:
                kb_text_parts.append(chunk_text[:remaining] + "\u2026")
            break
        kb_text_parts.append(chunk_text)
        total += len(chunk_text)

    kb_context = "\n\n---\n\n".join(kb_text_parts)
    prompt = RAG_PROMPT_TEMPLATE.format(
        kb_context=kb_context,
        conversation_history=conversation_history,
        user_query=user_query,
    )

    # Find the character span of the KB content within the assembled prompt.
    # The template places KB content immediately after "[KNOWLEDGE BASE]\n".
    kb_marker = "[KNOWLEDGE BASE]\n"
    section_start = prompt.find(kb_marker)
    if section_start == -1:
        return prompt, 0, 0
    kb_content_start = section_start + len(kb_marker)
    # Content ends just before "\n[CONVERSATION]"
    conv_marker = "\n[CONVERSATION]"
    kb_content_end = prompt.find(conv_marker, kb_content_start)
    if kb_content_end == -1:
        kb_content_end = kb_content_start + len(kb_context)

    return prompt, kb_content_start, kb_content_end


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic complexity (dry-run fallback — no GPU/LLM required)
# ─────────────────────────────────────────────────────────────────────────────

def heuristic_complexity(
    user_query: str,
    bot_response: str,
    kb_chunks: List[Tuple[str, float]],
) -> Dict[str, float]:
    """
    Estimate response complexity without a loaded LLM.

    Uses textual signals that correlate with deeper reasoning:
      - Response length (longer → more likely to be non-trivial)
      - Numeric content (numbers → factual precision required)
      - Conditional language (if/unless/depends → nuanced reasoning)
      - Query complexity (multi-clause questions)
      - KB retrieval similarity variance (diverse sources → harder synthesis)

    Returns a dict with keys matching TurnComplexityResult DTS fields.
    """
    resp_words   = bot_response.split()
    resp_len     = len(resp_words)

    # Signal: numeric tokens in response
    num_nums = len(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", bot_response))

    # Signal: conditional / comparative language
    conditional_kw = ["if", "unless", "however", "but", "except", "depends",
                      "although", "otherwise", "provided", "only if"]
    num_conditional = sum(bot_response.lower().count(kw) for kw in conditional_kw)

    # Signal: query clause count (commas + question marks)
    query_complexity = user_query.count(",") + user_query.count("?")

    # Signal: KB similarity spread (high variance → model must reconcile sources)
    if kb_chunks:
        sims = [s for _, s in kb_chunks]
        sim_spread = float(np.std(sims)) if len(sims) > 1 else 0.0
        top_sim    = max(sims)
    else:
        sim_spread = 0.0
        top_sim    = 0.0

    # Combine signals into a normalised DTS proxy (0–1)
    raw = (
        min(resp_len / 100.0, 1.0) * 0.30
        + min(num_nums / 5.0, 1.0) * 0.25
        + min(num_conditional / 4.0, 1.0) * 0.20
        + min(query_complexity / 3.0, 1.0) * 0.15
        + sim_spread * 0.10
    )
    dts_proxy = float(np.clip(raw, 0.0, 1.0))

    # Approximate settling depth as fraction (deep signals → later settling)
    settling_frac = dts_proxy * 0.8 + 0.05   # range ~0.05–0.85

    # Deep-thinking tokens = tokens with high complexity contribution
    deep_tokens = int(dts_proxy * resp_len)

    return {
        "dts": round(dts_proxy, 4),
        "avg_settling_depth": round(settling_frac * 32, 2),  # assume 32-layer model
        "settling_depth_fraction": round(settling_frac, 4),
        "num_deep_thinking_tokens": deep_tokens,
        "total_tokens_analyzed": resp_len,
        "num_layers": 32,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main analyser class
# ─────────────────────────────────────────────────────────────────────────────

class DTSGoldenDatasetAnalyzer:
    """
    End-to-end pipeline:
      Golden dataset  →  turn parsing  →  KB retrieval  →  DTS analysis  →  CSV
    """

    def __init__(
        self,
        model_name_or_path: Optional[str],
        kb_path: str,
        dts_config: DTSConfig,
        device: str = "cpu",
        dry_run: bool = False,
        retrieval_top_k: int = 5,
        retrieval_min_sim: float = 0.25,
        max_kb_chars: int = 3000,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        attn_impl: str = "sdpa",
        compute_kb_attention: bool = True,
    ):
        self.dry_run             = dry_run
        self.dts_config          = dts_config
        self.device              = device
        self.retrieval_top_k     = retrieval_top_k
        self.retrieval_min_sim   = retrieval_min_sim
        self.max_kb_chars        = max_kb_chars
        self.load_in_4bit        = load_in_4bit
        self.load_in_8bit        = load_in_8bit
        self.attn_impl           = attn_impl
        # KAM analysis is skipped in dry-run or if user opts out
        self.compute_kb_attention = compute_kb_attention and not dry_run

        # ── Load KB retriever ─────────────────────────────────────────────────
        logger.info("Initialising KB retriever…")
        self.kb_retriever = LocalKBRetriever(
            kb_path=kb_path,
            device=device,
        )

        # ── Load generation model (skipped in dry-run) ────────────────────────
        self.dts_analyzer: Optional[DeepThinkingAnalyzer] = None
        self.tokenizer = None

        if not dry_run and model_name_or_path:
            self._load_model(model_name_or_path)
        elif not dry_run:
            logger.warning(
                "No model specified and --dry_run not set.  "
                "Falling back to heuristic complexity."
            )
            self.dry_run = True

    def _load_model(self, model_name_or_path: str):
        """Load causal LM + tokenizer and wrap in DeepThinkingAnalyzer."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            logger.info(f"Loading tokenizer: {model_name_or_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # ── Quantisation config ───────────────────────────────────────────
            # 4-bit NF4 keeps lm_head + layer-norms in bf16 so hidden-state
            # projections remain numerically stable for DTS computation.
            bnb_config = None
            if self.load_in_4bit and self.device != "cpu":
                logger.info("Using 4-bit NF4 quantisation (bitsandbytes)")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
            elif self.load_in_8bit and self.device != "cpu":
                logger.info("Using 8-bit quantisation (bitsandbytes)")
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)

            logger.info(f"Loading model: {model_name_or_path}  (device={self.device}, attn_impl={self.attn_impl})")
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                quantization_config=bnb_config,
                torch_dtype=(
                    torch.float32 if self.device == "cpu"
                    else torch.bfloat16 if bnb_config else torch.float16
                ),
                device_map="auto" if self.device != "cpu" else None,
                low_cpu_mem_usage=True,
                attn_implementation=self.attn_impl,
            )
            if self.device == "cpu":
                model = model.to(self.device)
            model.eval()

            # Resolve effective device after device_map="auto"
            effective_device = next(model.parameters()).device
            logger.info(f"Model on device: {effective_device}")

            self.tokenizer    = tokenizer
            self.dts_analyzer = DeepThinkingAnalyzer(
                model=model,
                tokenizer=tokenizer,
                config=self.dts_config,
                device=str(effective_device),
            )
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.warning("Falling back to heuristic complexity.")
            self.dry_run = True

    # ── Per-turn analysis ─────────────────────────────────────────────────────

    def analyze_turn(
        self,
        turn: ConversationTurn,
        conversation_id: Any,
        ground_truth_accuracy: int,
    ) -> TurnComplexityResult:
        """Analyze a single conversation turn and return its complexity result."""

        # 1. KB retrieval — query = user question + first sentence of bot response
        kb_query = turn.user_query
        if turn.bot_response:
            first_sent = turn.bot_response.split(".")[0]
            kb_query = f"{kb_query} {first_sent}"

        kb_chunks = self.kb_retriever.retrieve(
            kb_query,
            top_k=self.retrieval_top_k,
            min_sim=self.retrieval_min_sim,
        )

        num_kb_chunks   = len(kb_chunks)
        kb_top_sim      = round(kb_chunks[0][1], 4) if kb_chunks else 0.0
        kb_chunks_prev  = (
            kb_chunks[0][0][:200].replace("\n", " ") + "…"
            if kb_chunks else ""
        )

        # 2. Build RAG-augmented prompt (with KB character offsets for attention analysis)
        rag_prompt, kb_char_start, kb_char_end = build_rag_prompt_with_offsets(
            user_query=turn.user_query,
            conversation_history=turn.history_before,
            kb_chunks=kb_chunks,
            max_kb_chars=self.max_kb_chars,
        )

        # 3. Compute DTS
        error_msg = ""
        if self.dry_run or self.dts_analyzer is None:
            # ── Heuristic path ────────────────────────────────────────────────
            metrics = heuristic_complexity(turn.user_query, turn.bot_response, kb_chunks)
            dts               = metrics["dts"]
            avg_settling      = metrics["avg_settling_depth"]
            settling_frac     = metrics["settling_depth_fraction"]
            num_deep_tokens   = metrics["num_deep_thinking_tokens"]
            total_tokens      = metrics["total_tokens_analyzed"]
            bot_len_tokens    = total_tokens  # same estimator
        else:
            # ── Forced-decode DTS path ────────────────────────────────────────
            try:
                analysis: PromptDTSAnalysis = self.dts_analyzer.analyze_forced_decode(
                    prompt=rag_prompt,
                    existing_response=turn.bot_response,
                )
                dts             = round(analysis.dts, 4)
                avg_settling    = round(analysis.avg_settling_depth, 4)
                settling_frac   = round(analysis.settling_depth_fraction, 4)
                num_deep_tokens = sum(1 for m in analysis.token_metrics if m.is_deep_thinking)
                total_tokens    = analysis.total_tokens
                bot_len_tokens  = total_tokens
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"[Conv {conversation_id} T{turn.turn_idx}] DTS failed: {e}")
                metrics = heuristic_complexity(turn.user_query, turn.bot_response, kb_chunks)
                dts             = metrics["dts"]
                avg_settling    = metrics["avg_settling_depth"]
                settling_frac   = metrics["settling_depth_fraction"]
                num_deep_tokens = metrics["num_deep_thinking_tokens"]
                total_tokens    = metrics["total_tokens_analyzed"]
                bot_len_tokens  = total_tokens

        # Tokenize separately for the raw token count if tokenizer is available
        if self.tokenizer and not self.dry_run:
            try:
                bot_enc = self.tokenizer(turn.bot_response, return_tensors="pt")
                bot_len_tokens = bot_enc["input_ids"].shape[1]
            except Exception:
                pass

        # 5. KB Attention Mass (KAM) — single extra forward pass with output_attentions
        kb_attn_mass       = 0.0
        kb_attn_early      = 0.0
        kb_attn_mid        = 0.0
        kb_attn_late       = 0.0
        kb_attn_no_bos     = 0.0
        kb_attn_entropy    = 0.0
        dts_x_inv_kam      = 0.0

        if self.compute_kb_attention and self.dts_analyzer is not None and kb_char_end > kb_char_start:
            try:
                kam: Optional[KBAttentionResult] = self.dts_analyzer.analyze_kb_attention(
                    prompt        = rag_prompt,
                    response      = turn.bot_response,
                    kb_char_start = kb_char_start,
                    kb_char_end   = kb_char_end,
                )
                if kam is not None:
                    kb_attn_mass    = kam.kam_mean
                    kb_attn_early   = kam.kam_early_layers
                    kb_attn_mid     = kam.kam_mid_layers
                    kb_attn_late    = kam.kam_late_layers
                    kb_attn_no_bos  = kam.kam_mean_no_bos
                    kb_attn_entropy = kam.attn_entropy_mean
                    # DTS * (1 – KAM): high = model reasoned deeply but ignored KB => FAIL signal
                    dts_x_inv_kam   = round(dts * (1.0 - kb_attn_mass), 4)
                    logger.debug(
                        f"  [Conv {conversation_id} T{turn.turn_idx}] "
                        f"KAM={kb_attn_mass:.3f} (mid={kb_attn_mid:.3f}) "
                        f"entropy={kb_attn_entropy:.2f} "
                        f"DTS×(1-KAM)={dts_x_inv_kam:.3f}  "
                        f"KB_tokens={kam.num_kb_tokens}"
                    )
            except Exception as e:
                logger.warning(
                    f"[Conv {conversation_id} T{turn.turn_idx}] KB attention failed: {e}"
                )

        # 5. Complexity category
        if dts < 0.3:
            category = "shallow"
        elif dts < 0.7:
            category = "moderate"
        else:
            category = "deep"

        return TurnComplexityResult(
            conversation_id       = conversation_id,
            turn_idx              = turn.turn_idx,
            user_query_preview    = (turn.user_query[:120] + "…") if len(turn.user_query) > 120 else turn.user_query,
            bot_response_preview  = (turn.bot_response[:200] + "…") if len(turn.bot_response) > 200 else turn.bot_response,
            bot_response_length_chars  = len(turn.bot_response),
            bot_response_length_tokens = bot_len_tokens,
            num_kb_chunks_retrieved    = num_kb_chunks,
            kb_top_similarity          = kb_top_sim,
            kb_chunks_preview          = kb_chunks_prev,
            dts                        = dts,
            avg_settling_depth         = avg_settling,
            settling_depth_fraction    = settling_frac,
            num_deep_thinking_tokens   = num_deep_tokens,
            total_tokens_analyzed      = total_tokens,
            complexity_category        = category,
            ground_truth_accuracy      = ground_truth_accuracy,
            ground_truth_label         = "PASS" if ground_truth_accuracy == 1 else "FAIL",
            error                      = error_msg,
            kb_attention_mass          = kb_attn_mass,
            kb_attention_mass_early    = kb_attn_early,
            kb_attention_mass_mid      = kb_attn_mid,
            kb_attention_mass_late     = kb_attn_late,
            kb_attention_mass_no_bos   = kb_attn_no_bos,
            kb_attention_entropy       = kb_attn_entropy,
            dts_x_inv_kam              = dts_x_inv_kam,
        )

    # ── Dataset-level entry point ─────────────────────────────────────────────

    def analyze_dataset(
        self,
        df: pd.DataFrame,
        output_csv: str,
        max_rows: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Iterate over all conversations in `df`, parse turns, run per-turn DTS,
        write a CSV, and return the results DataFrame.

        Expected columns in `df`: '#'  (conv id),  'Transcript',  'Accuracy'
        """
        if max_rows:
            df = df.head(max_rows).copy()

        results: List[TurnComplexityResult] = []
        total_convs = len(df)

        for row_idx, row in df.iterrows():
            conv_id  = row.get("#", row.get("conversation_id", row_idx))
            transcript = str(row.get("Transcript", "")).strip()
            accuracy_val = row.get("Accuracy", -1)

            if pd.notna(accuracy_val):
                gt_accuracy = int(accuracy_val)
            else:
                # Fallback for alternative column names
                gt_str = str(row.get("Is Bot Correct", "")).strip().lower()
                gt_accuracy = 1 if gt_str in ("yes", "y", "true", "1", "correct") else 0

            logger.info(
                f"[{row_idx+1}/{total_convs}] Conv {conv_id}  "
                f"GT={'PASS' if gt_accuracy==1 else 'FAIL'}"
            )

            turns = parse_transcript_turns(transcript)
            if not turns:
                logger.warning(f"  No turns parsed for conversation {conv_id}")
                continue

            for turn in turns:
                if not turn.bot_response.strip():
                    logger.debug(f"  Turn {turn.turn_idx}: empty bot response, skipping DTS")
                    # Still record with zeroed metrics
                    results.append(TurnComplexityResult(
                        conversation_id=conv_id,
                        turn_idx=turn.turn_idx,
                        user_query_preview=turn.user_query[:120],
                        bot_response_preview="(no response)",
                        bot_response_length_chars=0,
                        bot_response_length_tokens=0,
                        num_kb_chunks_retrieved=0,
                        kb_top_similarity=0.0,
                        kb_chunks_preview="",
                        dts=0.0,
                        avg_settling_depth=0.0,
                        settling_depth_fraction=0.0,
                        num_deep_thinking_tokens=0,
                        total_tokens_analyzed=0,
                        complexity_category="shallow",
                        ground_truth_accuracy=gt_accuracy,
                        ground_truth_label="PASS" if gt_accuracy == 1 else "FAIL",
                    ))
                    continue

                t0 = time.time()
                result = self.analyze_turn(
                    turn=turn,
                    conversation_id=conv_id,
                    ground_truth_accuracy=gt_accuracy,
                )
                elapsed = time.time() - t0
                logger.info(
                    f"  T{turn.turn_idx}: DTS={result.dts:.3f} "
                    f"({result.complexity_category})  KB_chunks={result.num_kb_chunks_retrieved} "
                    f" [{elapsed:.1f}s]"
                )
                results.append(result)

        # Convert to DataFrame
        records = [asdict(r) for r in results]
        out_df = pd.DataFrame(records)

        # Save CSV
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        logger.info(f"\nSaved {len(out_df)} turn-level rows to: {out_path}")

        # Print summary
        self._print_summary(out_df)
        return out_df

    # ── Summary ───────────────────────────────────────────────────────────────

    @staticmethod
    def _print_summary(df: pd.DataFrame):
        """Print a readable summary to stdout."""
        sep = "=" * 70
        print(f"\n{sep}")
        print("DTS COMPLEXITY ANALYSIS — SUMMARY")
        print(sep)
        print(f"  Total turns analysed      : {len(df)}")
        print(f"  Unique conversations       : {df['conversation_id'].nunique()}")
        print(f"  Avg turns per conversation : {len(df)/df['conversation_id'].nunique():.1f}")
        print()
        print(f"  DTS (mean  ± std)  : {df['dts'].mean():.3f} ± {df['dts'].std():.3f}")
        print(f"  DTS median         : {df['dts'].median():.3f}")
        print(f"  DTS range          : [{df['dts'].min():.3f}, {df['dts'].max():.3f}]")
        print()
        cats = df["complexity_category"].value_counts()
        for cat in ["shallow", "moderate", "deep"]:
            count = cats.get(cat, 0)
            pct   = 100 * count / len(df) if len(df) else 0
            print(f"  {cat:10s} turns : {count:4d}  ({pct:.1f}%)")
        print()
        print("  DTS by ground-truth label:")
        for label in ["PASS", "FAIL"]:
            sub = df[df["ground_truth_label"] == label]
            if len(sub):
                print(f"    {label}: mean DTS = {sub['dts'].mean():.3f}  (n={len(sub)})")
        print()
        print("  Avg KB chunks retrieved    :", round(df["num_kb_chunks_retrieved"].mean(), 2))
        print("  Avg KB top similarity      :", round(df["kb_top_similarity"].mean(), 3))
        print()
        if df["kb_attention_mass"].gt(0).any():
            print("  KB Attention Mass (KAM) — grounding in KB chunks:")
            print(f"    Overall KAM (mean)         : {df['kb_attention_mass'].mean():.3f}")
            print(f"    KAM early layers (syntax)  : {df['kb_attention_mass_early'].mean():.3f}")
            print(f"    KAM mid   layers (semantic): {df['kb_attention_mass_mid'].mean():.3f}")
            print(f"    KAM late  layers (copy)    : {df['kb_attention_mass_late'].mean():.3f}")
            print(f"    KAM no-BOS (corrected)     : {df['kb_attention_mass_no_bos'].mean():.3f}")
            print(f"    Attn entropy (mean)        : {df['kb_attention_entropy'].mean():.3f}")
            print()
            print("  DTS × (1–KAM) — deep but KB-ignoring (FAIL predictor):")
            for label in ["PASS", "FAIL"]:
                sub = df[df["ground_truth_label"] == label]
                if len(sub):
                    print(
                        f"    {label}: KAM={sub['kb_attention_mass'].mean():.3f}  "
                        f"DTS×(1-KAM)={sub['dts_x_inv_kam'].mean():.3f}  "
                        f"(n={len(sub)})"
                    )
        print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="DTS complexity analysis on the golden dataset (turn-wise, RAG-aware)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--excel_file",          default="Golden_dataset_100.xlsx",
                   help="Path to the golden dataset Excel file")
    p.add_argument("--output_csv",          default="dts_complexity_results.csv",
                   help="Output CSV file path")
    p.add_argument("--model_name_or_path",  default=None,
                   help="HuggingFace model id or local path (e.g. meta-llama/Llama-2-7b-chat-hf)")
    p.add_argument("--kb_path",             default=KB_DEFAULT_PATH,
                   help="Path to local KB Markdown directory")
    p.add_argument("--device",              default="cpu",
                   choices=["cpu", "cuda", "mps"],
                   help="Compute device")
    p.add_argument("--dry_run",             action="store_true",
                   help="Use heuristic complexity — no LLM loaded")
    p.add_argument("--max_rows",            type=int, default=None,
                   help="Limit number of conversations processed (for testing)")
    p.add_argument("--retrieval_top_k",     type=int, default=5,
                   help="Number of KB chunks to retrieve per turn")
    p.add_argument("--retrieval_min_sim",   type=float, default=0.25,
                   help="Minimum cosine similarity for KB retrieval")
    p.add_argument("--max_kb_chars",        type=int, default=3000,
                   help="Max characters of KB context included in prompt")
    # DTS hyperparams
    p.add_argument("--settling_threshold",  type=float, default=0.5,
                   help="JSD threshold below which a token is considered 'settled' (g)")
    p.add_argument("--depth_fraction",      type=float, default=0.85,
                   help="Layer-fraction ρ: tokens settling after this are 'deep-thinking'")
    p.add_argument("--load_in_4bit",        action="store_true",
                   help="Load model in 4-bit NF4 (bitsandbytes) — fits 8B in ~6 GB VRAM")
    p.add_argument("--load_in_8bit",        action="store_true",
                   help="Load model in 8-bit (bitsandbytes) — fits 8B in ~10 GB VRAM")
    p.add_argument("--attn_impl",           default="sdpa",
                   choices=["sdpa", "eager", "flash_attention_2"],
                   help="Attention implementation. 'sdpa' and 'eager' support output_attentions "
                        "(required for KB attention analysis). 'flash_attention_2' disables KAM.")
    p.add_argument("--no_kb_attention",     action="store_true",
                   help="Skip KB attention (KAM) analysis — faster but no grounding metrics")
    p.add_argument("--verbose",             action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Load dataset ──────────────────────────────────────────────────────────
    if not Path(args.excel_file).exists():
        logger.error(f"Dataset not found: {args.excel_file}")
        sys.exit(1)

    logger.info(f"Loading golden dataset: {args.excel_file}")
    df = pd.read_excel(args.excel_file)
    logger.info(f"  {len(df)} conversations, columns: {list(df.columns)}")

    # ── DTS config ────────────────────────────────────────────────────────────
    dts_config = DTSConfig(
        settling_threshold=args.settling_threshold,
        depth_fraction=args.depth_fraction,
        device=args.device,
        verbose=args.verbose,
    )

    # ── Analyser ──────────────────────────────────────────────────────────────
    analyser = DTSGoldenDatasetAnalyzer(
        model_name_or_path=args.model_name_or_path,
        kb_path=args.kb_path,
        dts_config=dts_config,
        device=args.device,
        dry_run=args.dry_run,
        retrieval_top_k=args.retrieval_top_k,
        retrieval_min_sim=args.retrieval_min_sim,
        max_kb_chars=args.max_kb_chars,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        attn_impl=args.attn_impl,
        compute_kb_attention=not args.no_kb_attention,
    )

    # ── Run ───────────────────────────────────────────────────────────────────
    out_df = analyser.analyze_dataset(
        df=df,
        output_csv=args.output_csv,
        max_rows=args.max_rows,
    )

    logger.info("Done.")
    return out_df


if __name__ == "__main__":
    main()

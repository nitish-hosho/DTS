"""
Deep Thinking Score (DTS) Analysis

Analyzes which prompts require deeper computational thinking in language models.
DTS measures how much a token's prediction evolves across model layers.

Key metrics:
- DTS (Deep Thinking Score): Proportion of tokens that settle in deep layers
- Settling Depth: First layer where prediction becomes stable
- Distributional Distance: Jensen-Shannon divergence from final prediction
"""

import gc
import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
from datetime import datetime
from scipy.spatial.distance import jensenshannon
import logging

logger = logging.getLogger(__name__)


@dataclass
class DTSConfig:
    """Configuration for DTS analysis"""
    settling_threshold: float = 0.5  # g in the paper
    depth_fraction: float = 0.85  # ρ: tokens settling after this fraction are "deep-thinking"
    extract_final_layer: bool = True
    batch_size: int = 1
    device: str = "cpu"
    verbose: bool = False


@dataclass
class TokenDTSMetrics:
    """Metrics for a single token"""
    token_id: int
    token_text: str
    settling_depth: int  # c_t: first layer where min distance <= threshold
    distances: List[float]  # D_t,l for each layer
    min_distances: List[float]  # Ḋ_t,l (running minimum)
    is_deep_thinking: bool
    final_logits: np.ndarray


@dataclass
class PromptDTSAnalysis:
    """Complete DTS analysis for a prompt"""
    prompt: str
    generated_text: str
    num_layers: int
    total_tokens: int
    token_metrics: List[TokenDTSMetrics] = field(default_factory=list)
    dts: float = 0.0  # Deep Thinking Score
    avg_settling_depth: float = 0.0
    settling_depth_fraction: float = 0.0  # As % of total layers
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        """Calculate aggregated metrics"""
        if self.token_metrics:
            deep_count = sum(1 for m in self.token_metrics if m.is_deep_thinking)
            self.dts = deep_count / len(self.token_metrics)
            self.avg_settling_depth = np.mean([m.settling_depth for m in self.token_metrics])
            self.settling_depth_fraction = self.avg_settling_depth / self.num_layers


@dataclass
class KBAttentionResult:
    """
    KB Attention Mass (KAM) metrics from a single forward pass over
    [prompt + response] with output_attentions=True.

    KAM measures what fraction of attention from response tokens flows
    to the KB-chunk token positions — i.e. how much the model grounded
    its answer in the retrieved knowledge base.

    High DTS + Low KAM => model reasoned deeply but ignored KB => FAIL signal.
    Low DTS + High KAM => easy copy from KB => reliable PASS signal.
    """
    kb_token_indices: List[int]           # token positions of KB section in full prompt
    num_kb_tokens: int                    # how many KB tokens in prompt
    num_response_tokens: int

    # Per-response-token KAM (mean over heads & layers), one float per response token
    kam_per_response_token: List[float]

    # Scalar aggregates (mean over response tokens, heads, layers)
    kam_mean: float          # overall KB attention mass (all layers)
    kam_early_layers: float  # layers 1 – floor(L/3)   : syntax / surface matching
    kam_mid_layers: float    # layers floor(L/3) – floor(2L/3) : semantic retrieval (key signal)
    kam_late_layers: float   # layers floor(2L/3) – L  : reasoning / copying

    # BOS-excluded: attention sink at position 0 dilutes KAM; this corrects that
    kam_mean_no_bos: float

    # Attention entropy over source positions at middle layer (mean over response tokens)
    # Low entropy + high KAM => confident single-chunk retrieval
    # High entropy + low KAM => model is "scanning" without clear KB match
    attn_entropy_mean: float


class LayerHookManager:
    """Manages hooks to extract intermediate representations from model layers"""

    def __init__(self, model, device: str = "cpu"):
        self.model = model
        self.device = device
        self.hooks = []
        self.intermediate_outputs = {}
        self.layer_mapping = {}  # Map module to layer index

    def create_hook(self, layer_idx: int):
        """Create a hook for a specific layer"""
        def hook(module, input, output):
            # Store the hidden state
            if isinstance(output, tuple):
                hidden_state = output[0]
            else:
                hidden_state = output

            # Save activation
            if layer_idx not in self.intermediate_outputs:
                self.intermediate_outputs[layer_idx] = []

            # Clone and detach to avoid memory issues
            self.intermediate_outputs[layer_idx].append(
                hidden_state.detach().cpu() if hidden_state.is_cuda else hidden_state.detach()
            )

        return hook

    def register_hooks(self) -> List[torch.utils.hooks.RemovableHandle]:
        """Register hooks to transformer layers intelligently"""
        handles = []
        self.intermediate_outputs = {}

        # Detect layer type and register hooks
        layer_idx = 0

        # For LLaMA/Mistral style models
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            for i, layer in enumerate(self.model.model.layers):
                hook_fn = self.create_hook(i)
                handle = layer.register_forward_hook(hook_fn)
                handles.append(handle)
                layer_idx = i + 1

        # For GPT2/GPT-style models
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            for i, layer in enumerate(self.model.transformer.h):
                hook_fn = self.create_hook(i)
                handle = layer.register_forward_hook(hook_fn)
                handles.append(handle)
                layer_idx = i + 1

        # Fallback: auto-detect transformer blocks
        if layer_idx == 0:
            for name, module in self.model.named_modules():
                if any(pattern in name for pattern in ['transformer.h', 'model.layers', 'decoder.layers']):
                    hook_fn = self.create_hook(layer_idx)
                    handle = module.register_forward_hook(hook_fn)
                    handles.append(handle)
                    layer_idx += 1

        self.hooks = handles
        logger.info(f"Registered {len(handles)} layer hooks")
        return handles

    def remove_hooks(self):
        """Remove all registered hooks"""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def reset(self):
        """Reset stored outputs"""
        self.intermediate_outputs = {}

    def get_outputs(self) -> Dict[int, List[Tensor]]:
        """Get all collected outputs"""
        return self.intermediate_outputs


class DeepThinkingAnalyzer:
    """Analyzes Deep Thinking Score for prompts"""

    def __init__(
        self,
        model,
        tokenizer,
        config: DTSConfig = None,
        device: str = "cpu"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or DTSConfig(device=device)
        self.device = device
        self.hook_manager = LayerHookManager(model, device)

        # Get unembedding matrix (final layer output projection)
        if hasattr(model, 'lm_head'):
            self.unembedding_matrix = model.lm_head.weight.data.cpu()
        elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            self.unembedding_matrix = None
        else:
            self.unembedding_matrix = None

        self.num_layers = self._detect_num_layers()

    def _detect_num_layers(self) -> int:
        """Detect number of transformer layers"""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return len(self.model.model.layers)
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return len(self.model.transformer.h)
        elif hasattr(self.model, 'config'):
            if hasattr(self.model.config, 'num_hidden_layers'):
                return self.model.config.num_hidden_layers
        return 32  # Default fallback

    def _get_output_projection(self) -> Optional[torch.nn.Module]:
        """Get the output projection layer that maps to vocabulary"""
        if hasattr(self.model, 'lm_head'):
            return self.model.lm_head
        if hasattr(self.model, 'head'):
            return self.model.head
        if hasattr(self.model, 'classifier'):
            return self.model.classifier
        return None

    def _compute_jsd(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute Jensen-Shannon Divergence between two probability distributions.

        JSD(P||Q) = H((P+Q)/2) - 0.5*H(P) - 0.5*H(Q)
        where H is Shannon entropy
        """
        try:
            p = np.asarray(p, dtype=np.float64)
            q = np.asarray(q, dtype=np.float64)

            p = np.nan_to_num(p, nan=1e-10, posinf=1.0, neginf=0.0)
            q = np.nan_to_num(q, nan=1e-10, posinf=1.0, neginf=0.0)

            p = np.clip(p, 1e-10, 1.0)
            q = np.clip(q, 1e-10, 1.0)

            p_sum = p.sum()
            q_sum = q.sum()

            if p_sum <= 0 or q_sum <= 0:
                return 1.0

            p = p / p_sum
            q = q / q_sum

            jsd_val = float(jensenshannon(p, q))
            return float(np.clip(jsd_val, 0.0, 1.0))
        except Exception as e:
            logger.debug(f"JSD calculation failed: {e}")
            return 1.0

    def _get_layer_distributions(
        self,
        layer_outputs: Dict[int, Tensor],
        final_logits: Tensor
    ) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
        """
        Compute normalized hidden state distributions from layer outputs.

        Returns: (Dict mapping layer_idx -> normalized state, final normalized logits)
        """
        distributions = {}

        final_probs = F.softmax(final_logits.float(), dim=-1).detach().cpu().numpy()
        final_probs_normalized = final_probs / (np.linalg.norm(final_probs) + 1e-10)

        for layer_idx, outputs in sorted(layer_outputs.items()):
            if outputs:
                try:
                    output = outputs[-1].detach().cpu().float()

                    if output.dim() > 1:
                        output = output[-1]

                    output_np = output.numpy() if isinstance(output, torch.Tensor) else output
                    output_norm = output_np / (np.linalg.norm(output_np) + 1e-10)

                    if output_norm.shape[0] != final_probs.shape[0]:
                        scale = final_probs.shape[0] / output_norm.shape[0]
                        expanded = np.repeat(output_norm, int(np.ceil(scale)))[:final_probs.shape[0]]
                        output_norm = expanded / (np.linalg.norm(expanded) + 1e-10)

                    distributions[layer_idx] = output_norm

                except Exception as e:
                    logger.debug(f"Layer {layer_idx}: failed to process ({e}). Using zeros.")
                    distributions[layer_idx] = np.zeros(final_probs.shape[0])

        return distributions, final_probs_normalized

    def _calculate_settling_depth(
        self,
        distances: List[float],
        threshold: float
    ) -> int:
        """
        Find settling depth: first layer where min_distance up to that layer <= threshold.

        Args:
            distances: JSD distances for each layer from final distribution
            threshold: Settling threshold (g)

        Returns:
            Layer index where token settles, or len(distances) if never settles
        """
        min_distances = []
        running_min = float('inf')

        for dist in distances:
            running_min = min(running_min, dist)
            min_distances.append(running_min)

        for layer_idx, min_dist in enumerate(min_distances):
            if min_dist <= threshold:
                return layer_idx

        return len(distances) - 1

    @torch.no_grad()
    def analyze_prompt(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7
    ) -> PromptDTSAnalysis:
        """
        Analyze a prompt for deep-thinking tokens via generation.

        Args:
            prompt: Input prompt text
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature

        Returns:
            PromptDTSAnalysis with full metrics
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs['input_ids']

        self.hook_manager.register_hooks()

        generated_tokens = []
        token_metrics_list = []

        _final_norm = (
            getattr(getattr(self.model, 'model', None), 'norm', None)
            or getattr(getattr(self.model, 'transformer', None), 'ln_f', None)
            or getattr(getattr(self.model, 'model', None), 'final_layernorm', None)
        )

        try:
            for gen_idx in range(max_new_tokens):
                self.hook_manager.reset()

                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        output_hidden_states=True,
                        return_dict=True
                    )

                # ── Extract only what we need, then immediately free the large tensors ──
                final_logits = outputs.logits[0, -1, :].detach()
                final_probs  = F.softmax(final_logits.float(), dim=-1).cpu().numpy()

                # Pull last-position hidden vector from every layer (tiny: [d_model])
                # and discard the full [1, seq_len, d_model] tensors right away.
                last_hidden_per_layer = []
                for hs in outputs.hidden_states[1:]:          # skip embedding layer (idx 0)
                    last_hidden_per_layer.append(hs[0, -1, :].detach().clone())
                del outputs  # free logits + all hidden state tensors from GPU

                next_token_logits = final_logits / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()
                generated_tokens.append(next_token_id)

                distances = []
                for last_hidden in last_hidden_per_layer:
                    try:
                        h = last_hidden
                        if _final_norm is not None:
                            h = _final_norm(h.unsqueeze(0)).squeeze(0)
                        if hasattr(self.model, 'lm_head'):
                            layer_logits = self.model.lm_head(h)
                            layer_probs = F.softmax(layer_logits.float(), dim=-1).detach().cpu().numpy()
                        else:
                            layer_probs = final_probs.copy()
                        jsd = self._compute_jsd(final_probs, layer_probs)
                        distances.append(jsd)
                    except Exception:
                        distances.append(0.0)
                del last_hidden_per_layer

                settling_depth = self._calculate_settling_depth(
                    distances, self.config.settling_threshold
                )
                num_transformer_layers = len(distances)
                deep_threshold = int(np.ceil(self.config.depth_fraction * num_transformer_layers))
                is_deep_thinking = settling_depth >= deep_threshold

                metrics = TokenDTSMetrics(
                    token_id=next_token_id,
                    token_text=self.tokenizer.decode([next_token_id]),
                    settling_depth=settling_depth,
                    distances=distances,
                    min_distances=self._compute_min_distances(distances),
                    is_deep_thinking=is_deep_thinking,
                    final_logits=final_probs,
                )
                token_metrics_list.append(metrics)

                input_ids = torch.cat([
                    input_ids,
                    torch.tensor([[next_token_id]], device=input_ids.device)
                ], dim=1)

                # Periodic GPU cache flush to prevent fragmentation
                if gen_idx % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        finally:
            self.hook_manager.remove_hooks()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        analysis = PromptDTSAnalysis(
            prompt=prompt,
            generated_text=generated_text,
            num_layers=self.num_layers,
            total_tokens=len(generated_tokens),
            token_metrics=token_metrics_list
        )

        return analysis

    @torch.no_grad()
    def analyze_forced_decode(
        self,
        prompt: str,
        existing_response: str,
    ) -> PromptDTSAnalysis:
        """
        Analyze DTS over an *existing* (pre-generated) response using teacher-forcing.

        Instead of sampling tokens, we feed the known bot response tokens one-by-one
        and compute the DTS at each response-token position.  This reflects how hard
        the model has to "think" to predict that exact response given the (RAG-augmented)
        prompt.

        Args:
            prompt: Full RAG-augmented prompt (system + KB chunks + conv history + user query)
            existing_response: The actual bot response text from the dataset

        Returns:
            PromptDTSAnalysis with per-response-token DTS metrics
        """
        # Tokenize
        prompt_enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        response_enc = self.tokenizer(
            existing_response,
            return_tensors="pt",
            add_special_tokens=False
        )

        # Resolve the actual primary device (handles device_map="auto" spreading)
        try:
            primary_device = next(self.model.parameters()).device
        except StopIteration:
            primary_device = torch.device(self.device)

        prompt_ids = prompt_enc['input_ids'].to(primary_device)          # [1, P]
        response_ids = response_enc['input_ids'].to(primary_device)      # [1, R]
        num_response_tokens = response_ids.shape[1]

        if num_response_tokens == 0:
            return PromptDTSAnalysis(
                prompt=prompt,
                generated_text=existing_response,
                num_layers=self.num_layers,
                total_tokens=0,
                token_metrics=[]
            )

        # Detect final layer-norm once
        _final_norm = (
            getattr(getattr(self.model, 'model', None), 'norm', None)
            or getattr(getattr(self.model, 'transformer', None), 'ln_f', None)
            or getattr(getattr(self.model, 'model', None), 'final_layernorm', None)
        )

        token_metrics_list = []

        # Teacher-forcing: for each response token position r, run a forward pass
        # on prompt_ids + response_ids[:r] and evaluate the prediction at position -1.
        # We start from position 0 (predicting the first response token from the prompt alone)
        # up to position R-1 (predicting the last response token).
        current_input = prompt_ids.clone()

        for r in range(num_response_tokens):
            target_token_id = response_ids[0, r].item()

            with torch.no_grad():
                outputs = self.model(
                    input_ids=current_input,
                    output_hidden_states=True,
                    return_dict=True
                )

            hidden_states = outputs.hidden_states  # tuple: (embed, layer1, ..., layerN)
            final_logits = outputs.logits[0, -1, :]
            final_probs = F.softmax(final_logits.float(), dim=-1).detach().cpu().numpy()

            distances: List[float] = []

            if len(hidden_states) > 1:
                for layer_idx in range(1, len(hidden_states)):
                    try:
                        last_hidden = hidden_states[layer_idx][0, -1, :]  # keep model dtype

                        if _final_norm is not None:
                            last_hidden = _final_norm(last_hidden.unsqueeze(0)).squeeze(0)

                        if hasattr(self.model, 'lm_head'):
                            layer_logits = self.model.lm_head(last_hidden)
                            layer_probs = F.softmax(layer_logits.float(), dim=-1).detach().cpu().numpy()
                        else:
                            layer_probs = final_probs.copy()

                        jsd = self._compute_jsd(final_probs, layer_probs)
                        distances.append(jsd)
                    except Exception:
                        distances.append(0.0)

            settling_depth = self._calculate_settling_depth(
                distances, self.config.settling_threshold
            )

            num_transformer_layers = len(distances) if distances else self.num_layers
            deep_threshold = int(np.ceil(self.config.depth_fraction * num_transformer_layers))
            is_deep_thinking = settling_depth >= deep_threshold

            metrics = TokenDTSMetrics(
                token_id=target_token_id,
                token_text=self.tokenizer.decode([target_token_id]),
                settling_depth=settling_depth,
                distances=distances,
                min_distances=self._compute_min_distances(distances),
                is_deep_thinking=is_deep_thinking,
                final_logits=final_probs
            )
            token_metrics_list.append(metrics)

            # Extend input with the known response token
            current_input = torch.cat(
                [current_input, response_ids[:, r:r+1].to(primary_device)],
                dim=1
            )

        analysis = PromptDTSAnalysis(
            prompt=prompt,
            generated_text=existing_response,
            num_layers=self.num_layers,
            total_tokens=num_response_tokens,
            token_metrics=token_metrics_list
        )

        return analysis

    @staticmethod
    def _compute_min_distances(distances: List[float]) -> List[float]:
        """Compute running minimum of distances"""
        min_distances = []
        running_min = float('inf')
        for dist in distances:
            running_min = min(running_min, dist)
            min_distances.append(running_min)
        return min_distances

    # ── KB Attention Analysis ─────────────────────────────────────────────────

    def _get_kb_token_indices(
        self,
        prompt: str,
        kb_char_start: int,
        kb_char_end: int,
        device,
    ) -> List[int]:
        """
        Map character offsets [kb_char_start, kb_char_end) within `prompt` to
        token indices in the tokenized prompt.

        Tokenizes the prefix and the KB section separately so BPE boundaries do
        not cause off-by-one errors.  The BOS token added by the full-prompt
        tokenization is accounted for via add_special_tokens=True on the prefix.
        """
        if kb_char_start >= kb_char_end:
            return []
        try:
            prefix_text = prompt[:kb_char_start]
            kb_text     = prompt[kb_char_start:kb_char_end]
            prefix_ids = self.tokenizer(prefix_text, add_special_tokens=True)["input_ids"]
            kb_ids     = self.tokenizer(kb_text, add_special_tokens=False)["input_ids"]
            prefix_len = len(prefix_ids)
            kb_len     = len(kb_ids)
            return list(range(prefix_len, prefix_len + kb_len))
        except Exception as e:
            logger.debug(f"_get_kb_token_indices failed: {e}")
            return []

    @torch.no_grad()
    def analyze_kb_attention(
        self,
        prompt: str,
        response: str,
        kb_char_start: int,
        kb_char_end: int,
    ) -> Optional["KBAttentionResult"]:
        """
        Compute KB Attention Mass (KAM): the fraction of attention from each
        response token that flows to the KB-chunk tokens in the prompt.

        A single forward pass is run over [full_prompt + response] with
        output_attentions=True.  This is separate from the per-token DTS passes
        so it adds only ~1 forward pass (vs R passes for DTS).

        Requires attn_implementation='sdpa' (PyTorch scaled-dot-product attention).
        Flash Attention 2 does NOT return attention weights and will yield None.

        Args:
            prompt: Full RAG-augmented prompt string (same as fed to DTS).
            response: Bot response string.
            kb_char_start: Character start of KB content within prompt.
            kb_char_end: Character end of KB content within prompt.

        Returns:
            KBAttentionResult or None if attention weights are unavailable.
        """
        try:
            primary_device = next(self.model.parameters()).device
        except StopIteration:
            primary_device = torch.device(self.device)

        # Tokenize prompt and response
        prompt_enc   = self.tokenizer(prompt,   return_tensors="pt", add_special_tokens=True)
        response_enc = self.tokenizer(response, return_tensors="pt", add_special_tokens=False)

        prompt_ids   = prompt_enc["input_ids"].to(primary_device)    # (1, P)
        response_ids = response_enc["input_ids"].to(primary_device)  # (1, R)

        P = prompt_ids.shape[1]
        R = response_ids.shape[1]
        if R == 0:
            return None

        # KB token positions in the full input sequence
        kb_token_indices = self._get_kb_token_indices(
            prompt, kb_char_start, kb_char_end, primary_device
        )
        if not kb_token_indices:
            logger.debug("No KB token indices found; skipping KB attention analysis.")
            return None

        # Single forward pass over [prompt + response]
        full_ids = torch.cat([prompt_ids, response_ids], dim=1)   # (1, P+R)
        try:
            outputs = self.model(
                input_ids=full_ids,
                output_attentions=True,
                return_dict=True,
            )
        except Exception as e:
            logger.warning(f"Forward pass with output_attentions failed: {e}")
            return None

        # outputs.attentions is a tuple of (1, H, seq, seq) per layer, or None for FA2
        if outputs.attentions is None or len(outputs.attentions) == 0:
            logger.warning(
                "output_attentions=None — model is likely using Flash Attention 2. "
                "Reload with attn_implementation='sdpa' to enable KB attention analysis."
            )
            return None

        L = len(outputs.attentions)
        early_end = max(1, L // 3)
        mid_end   = max(early_end + 1, (2 * L) // 3)

        kb_set        = set(kb_token_indices)
        kb_set_no_bos = {i for i in kb_set if i != 0}  # exclude BOS attention sink
        kb_list       = sorted(kb_set)
        kb_list_nobos = sorted(kb_set_no_bos)

        # Response token positions in the concatenated sequence
        response_positions = list(range(P, P + R))

        # layer_kams[l, r] = mean-over-heads attention from response token r to all KB tokens
        layer_kams       = np.zeros((L, R), dtype=np.float32)
        layer_kams_nobos = np.zeros((L, R), dtype=np.float32)

        for l_idx, attn_tensor in enumerate(outputs.attentions):
            # attn_tensor: (1, H, seq, seq); keep float32 for numpy
            attn_np = attn_tensor[0].float().detach().cpu().numpy()  # (H, seq, seq)
            for r_i, r_pos in enumerate(response_positions):
                a_h = attn_np[:, r_pos, :]  # (H, P+R)
                # KAM: mean over heads of sum-of-attention to KB positions
                layer_kams[l_idx, r_i] = a_h[:, kb_list].sum(axis=1).mean()
                if kb_list_nobos:
                    # BOS-excluded: renormalize by total non-BOS attention
                    non_bos = [i for i in range(full_ids.shape[1]) if i != 0]
                    denom   = a_h[:, non_bos].sum(axis=1).mean() + 1e-9
                    layer_kams_nobos[l_idx, r_i] = (
                        a_h[:, kb_list_nobos].sum(axis=1).mean() / denom
                    )

        # Scalar aggregates
        kam_mean  = float(layer_kams.mean())
        kam_early = float(layer_kams[:early_end].mean())
        kam_mid   = float(layer_kams[early_end:mid_end].mean())
        kam_late  = float(layer_kams[mid_end:].mean())
        kam_nobos = float(layer_kams_nobos.mean())

        # Per-response-token KAM: mean over all layers
        kam_per_token = list(layer_kams.mean(axis=0).astype(float))

        # Attention entropy at the middle layer (mean over response tokens)
        mid_layer_idx = L // 2
        mid_attn_np   = outputs.attentions[mid_layer_idx][0].float().detach().cpu().numpy()
        entropies = []
        for r_pos in response_positions:
            a_mean = mid_attn_np[:, r_pos, :P].mean(axis=0)  # only attend over prompt (causal)
            a_mean = np.clip(a_mean, 1e-10, None)
            a_mean = a_mean / (a_mean.sum() + 1e-10)
            entropies.append(float(-np.sum(a_mean * np.log(a_mean))))

        return KBAttentionResult(
            kb_token_indices    = kb_token_indices,
            num_kb_tokens       = len(kb_token_indices),
            num_response_tokens = R,
            kam_per_response_token = kam_per_token,
            kam_mean            = round(kam_mean,  4),
            kam_early_layers    = round(kam_early, 4),
            kam_mid_layers      = round(kam_mid,   4),
            kam_late_layers     = round(kam_late,  4),
            kam_mean_no_bos     = round(kam_nobos, 4),
            attn_entropy_mean   = round(float(np.mean(entropies)), 4),
        )

    def analyze_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 50
    ) -> List[PromptDTSAnalysis]:
        """Analyze multiple prompts"""
        results = []
        for i, prompt in enumerate(prompts):
            logger.info(f"Analyzing prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
            analysis = self.analyze_prompt(prompt, max_new_tokens)
            results.append(analysis)
        return results

    def save_analysis(self, analysis: PromptDTSAnalysis, output_path: Path):
        """Save analysis results to JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'prompt': analysis.prompt,
            'generated_text': analysis.generated_text,
            'num_layers': analysis.num_layers,
            'total_tokens': analysis.total_tokens,
            'dts': float(analysis.dts),
            'avg_settling_depth': float(analysis.avg_settling_depth),
            'settling_depth_fraction': float(analysis.settling_depth_fraction),
            'timestamp': analysis.timestamp,
            'token_metrics': [
                {
                    'token_id': m.token_id,
                    'token_text': m.token_text,
                    'settling_depth': m.settling_depth,
                    'distances': m.distances,
                    'min_distances': m.min_distances,
                    'is_deep_thinking': m.is_deep_thinking,
                }
                for m in analysis.token_metrics
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Analysis saved to {output_path}")


class DTSComparator:
    """Compare DTS across multiple prompts to identify thinking patterns"""

    @staticmethod
    def categorize_prompts(analyses: List[PromptDTSAnalysis]) -> Dict[str, List[PromptDTSAnalysis]]:
        """Categorize prompts by thinking depth"""
        categories = {
            'shallow': [],   # DTS < 0.3
            'moderate': [],  # 0.3 <= DTS < 0.7
            'deep': []       # DTS >= 0.7
        }

        for analysis in analyses:
            if analysis.dts < 0.3:
                categories['shallow'].append(analysis)
            elif analysis.dts < 0.7:
                categories['moderate'].append(analysis)
            else:
                categories['deep'].append(analysis)

        return categories

    @staticmethod
    def generate_report(analyses: List[PromptDTSAnalysis]) -> Dict[str, Any]:
        """Generate summary report"""
        if not analyses:
            return {}

        dtrs = [a.dts for a in analyses]
        settling_depths = [a.settling_depth_fraction for a in analyses]

        categories = DTSComparator.categorize_prompts(analyses)

        return {
            'total_prompts': len(analyses),
            'avg_dtr': float(np.mean(dtrs)),
            'median_dtr': float(np.median(dtrs)),
            'std_dtr': float(np.std(dtrs)),
            'min_dtr': float(np.min(dtrs)),
            'max_dtr': float(np.max(dtrs)),
            'avg_settling_depth_fraction': float(np.mean(settling_depths)),
            'category_breakdown': {
                category: len(prompts_list)
                for category, prompts_list in categories.items()
            },
            'deep_thinking_prompts': [
                {
                    'prompt': a.prompt,
                    'dts': a.dts,
                    'settling_depth': a.settling_depth_fraction
                }
                for a in sorted(analyses, key=lambda x: x.dts, reverse=True)[:10]
            ]
        }

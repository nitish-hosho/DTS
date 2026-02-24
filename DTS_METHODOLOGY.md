# Deep Thinking Score (DTS) — Methodology Note

## What It Measures

**DTS (Deep Thinking Score)** quantifies the fraction of response tokens whose
final prediction is not resolved until the deepest layers of a transformer, treating
layer-wise prediction settling as a proxy for *reasoning difficulty/answering diffculty*.

A high DTS means the model needed sustained, deep-layer computation (many layers
still "changing their mind") to produce the response — i.e. the response was hard
to reason about.  A low DTS means shallow, early layers already locked in the
prediction — the response was routine retrieval or pattern completion.

---

## Mathematical Formulation

### 1. Per-layer vocabulary projection  

For response token at position $t$, the hidden state at layer $l$ is
$\hat{h}_{t,l} \in \mathbb{R}^d$.  The layer's implied probability distribution
over the vocabulary is:

$$P_{t,l} = \text{softmax}\!\left(\mathbf{W}_U\, \hat{h}_{t,l}\right)$$

where $\mathbf{W}_U$ is the unembedding (lm_head) weight matrix.  
$P_{t,L}$ is the distribution at the final layer $L$.

### 2. Jensen–Shannon Divergence from final prediction  

The distributional distance of layer $l$ from the final prediction is:

$$D_{t,l} = \operatorname{JSD}\!\left(P_{t,l} \;\|\; P_{t,L}\right)
           = H\!\left(\tfrac{P_{t,l}+P_{t,L}}{2}\right)
             - \tfrac{1}{2}H(P_{t,l}) - \tfrac{1}{2}H(P_{t,L})$$

where $H(\cdot)$ is Shannon entropy.  $D_{t,l} \in [0,\,1]$ with
$D_{t,L}=0$ by definition.

### 3. Running minimum (cumulative settling signal)  

$$\dot{D}_{t,l} = \min_{l' \le l} D_{t,l'}$$

$\dot{D}_{t,l}$ is monotonically non-increasing — once the distribution
converges it cannot diverge again.

### 4. Settling depth  

Given settling threshold $g$ (default 0.5):

$$c_t = \min\!\bigl\{\, l : \dot{D}_{t,l} \le g \,\bigr\}$$

$c_t$ is the first layer at which the running minimum falls below $g$.
A large $c_t$ means the prediction remained uncertain deep into the network.

### 5. Deep-thinking indicator  

Given depth fraction $\rho$ (default 0.85):

$$\mathbf{1}_{\text{deep}}(t) =
  \begin{cases}
    1 & \text{if } c_t \ge \lceil \rho \cdot L \rceil \\
    0 & \text{otherwise}
  \end{cases}$$

A token is *deep-thinking* if it settles only in the last $(1{-}\rho)\cdot 100\%$
of layers.  For LLaMA-3.1-8B ($L=32$, $\rho=0.85$): threshold layer = 28.

### 6. Deep Thinking Score (DTS)  

Over all $N$ tokens in the forced-decoded response:

$$\boxed{\text{DTS} = \dfrac{1}{N} \sum_{t=1}^{N} \mathbf{1}_{\text{deep}}(t)}$$

DTS $\in [0, 1]$.  Calibrated categories (empirical, LLaMA-3.1-8B):

| DTS range | Category |
|-----------|----------|
| < 0.3 | Shallow — early retrieval, template completion |
| 0.3 – 0.7 | Moderate — some policy/conditional reasoning |
| ≥ 0.7 | Deep — multi-step inference, complex policy chains |

### 7. Auxiliary metrics  

**Average settling depth:**

$$\bar{c} = \frac{1}{N}\sum_{t=1}^{N} c_t$$

**Settling depth fraction (SDF):**

$$\text{SDF} = \bar{c} / L$$

SDF expresses the typical settling layer as a fraction of the model depth,
making it comparable across models with different layer counts.

---

## Implementation Details

### Teacher-forcing (forced-decode)

Rather than letting the model generate freely, the pipeline feeds the *known*
bot response token-by-token (teacher-forcing) while collecting hidden states at
every layer.  This avoids sampling noise and ensures DTS reflects the
complexity of the actual observed response under the given context.

```
prompt_ids = tokenize(rag_prompt)           # [SYSTEM]+[KB]+[HISTORY]+User:query\nBot:
response_ids = tokenize(bot_response)

for r in range(len(response_ids)):
    input = concat(prompt_ids, response_ids[:r+1])
    run forward pass → collect hidden_states[0..L]
    compute P_{r,l} for each l, then D_{r,l}, then c_r
    accumulate 1_deep(r)

DTS = sum(1_deep) / len(response_ids)
```

### RAG-augmented prompt structure

```
[SYSTEM]
You are an airline customer-support assistant. Answer based ONLY on the
knowledge base excerpts below.

[KNOWLEDGE BASE]
<top-3 KB chunks retrieved by cosine similarity to user query>

[CONVERSATION]
<prior turns in User:/Bot: format>
User: <current query>
Bot:
```

KB retrieval uses `sentence-transformers/all-MiniLM-L6-v2` with cosine
similarity; top-3 chunks with min similarity 0.25 are included.

---

## Empirical Results — Golden Dataset (LLaMA-3.1-8B-Instruct, 4-bit NF4)

| Split | n turns | Mean DTS | Std |
|-------|---------|----------|-----|
| PASS (Accuracy=1) | 115 | **0.490** | — |
| FAIL (Accuracy=0) | 56 | **0.592** | — |
| Overall | 171 | — | — |

**Key finding:** FAIL conversations exhibit ~10 percentage-point higher DTS on
average, consistent with the interpretation that *harder-to-answer* queries
(where the bot made errors) require deeper network computation.  High DTS + FAIL
label = "effortful but wrong" — the model engaged deep reasoning yet still
produced an incorrect response.

| Complexity bucket | Turns |
|-------------------|-------|
| Shallow (DTS < 0.3) | 96 (56.1%) |
| Moderate (0.3 ≤ DTS < 0.7) | 38 (22.2%) |
| Deep (DTS ≥ 0.7) | 37 (21.6%) |

---

## Key Hyperparameters

| Parameter | Symbol | Default | Effect |
|-----------|--------|---------|--------|
| Settling threshold | $g$ | 0.5 | Higher → more tokens counted as "deep" |
| Depth fraction | $\rho$ | 0.85 | Higher → only last few layers count as "deep" |
| Retrieval top-k | — | 3 | KB chunks included in prompt |
| Min KB similarity | — | 0.25 | Cosine cutoff for relevance |

---

## Files

| File | Role |
|------|------|
| `dtr_analyzer.py` | Core DTS engine — `DTSConfig`, `TokenDTSMetrics`, `PromptDTSAnalysis`, `DeepThinkingAnalyzer`, `DTSComparator` |
| `dtr_complexity_golden_dataset.py` | End-to-end pipeline — KB retrieval, turn parsing, teacher-forcing, CSV export |
| `dts_complexity_results_llama8b.csv` | Turn-wise DTS output for Golden dataset (171 turns, LLaMA-3.1-8B-Instruct 4-bit) |
| `dts_kb_embed_cache.pkl` | Cached MiniLM embeddings for KB chunks |

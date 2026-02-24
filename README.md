# Deep Thinking Score (DTS) — Scoring Service

Score any prompt for **reasoning depth** with one API call.  
Send a prompt → get back the **DTS score** and exactly **which tokens needed deep thinking**.

See [DTS_METHODOLOGY.md](DTS_METHODOLOGY.md) for the full mathematical formulation
and empirical results.

---

## How it works

1. The service loads **LLaMA-3.1-8B-Instruct** (NF4 4-bit) once at startup.
2. `POST /score` receives your prompt, generates a response, then runs a
   layer-wise JSD analysis over every generated token.
3. Returns the **DTS score** + a list of tokens that settled only in the
   deepest layers of the network (the "hard" tokens).

---

## Repository layout

```
dts_methodology/
├── main.py                           # FastAPI application (start here)
├── model_loader.py                   # Singleton model / analyzer loader
├── schemas.py                        # Pydantic request & response models
├── dtr_analyzer.py                   # Core DTS engine
├── dtr_complexity_golden_dataset.py  # Offline batch pipeline (RAG + teacher-forcing)
├── DTS_METHODOLOGY.md                # Methodology note (math, results, hyperparameters)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick start

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> GPU with ≥16 GB VRAM recommended (LLaMA-3.1-8B in NF4 uses ~6 GB).  
> CPU inference supported but slow.

### 2. Configure environment

```bash
cp .env.example .env
# edit .env — set HUGGING_FACE_HUB_TOKEN for gated LLaMA weights
```

### 3. Start the service

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Interactive API docs → **http://localhost:8000/docs**

---

## API

### `POST /score`

**Request body**

```json
{
  "prompt": "Explain the difference between supervised and unsupervised learning.",
  "max_new_tokens": 100,
  "settling_threshold": 0.5,
  "depth_fraction": 0.85,
  "include_jsd_trace": false
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | **required** | Text to score |
| `max_new_tokens` | int | 100 | Response tokens to generate |
| `settling_threshold` | float | 0.5 | JSD threshold `g` |
| `depth_fraction` | float | 0.85 | Layer depth fraction `ρ` |
| `include_jsd_trace` | bool | false | Include per-layer JSD for every token |

**Response**

```json
{
  "dts_score": 0.7292,
  "category": "Deep",
  "generated_response": "Supervised learning uses labelled data …",
  "total_tokens": 48,
  "avg_settling_depth": 27.6,
  "settling_depth_fraction": 0.8625,
  "num_deep_tokens": 35,
  "deep_tokens": [
    {
      "position": 2,
      "token": " labelled",
      "token_id": 30383,
      "settling_layer": 29,
      "settling_depth_fraction": 0.9063,
      "is_deep": true,
      "jsd_trace": null
    }
  ],
  "all_tokens": [ … ],
  "model_name": "meta-llama/Llama-3.1-8B-Instruct",
  "num_layers": 32,
  "deep_layer_threshold": 28,
  "settling_threshold": 0.5,
  "depth_fraction": 0.85
}
```

| Field | Description |
|-------|-------------|
| `dts_score` | Deep Thinking Score ∈ [0, 1] |
| `category` | `Shallow` (<0.3) · `Moderate` (0.3–0.7) · `Deep` (≥0.7) |
| `deep_tokens` | Tokens that settled only in the last (1−ρ)·100 % of layers |
| `all_tokens` | Every generated token with full per-token metrics |

### `GET /health`

Returns model load status, device, and layer count.

---

## DTS categories (LLaMA-3.1-8B empirical calibration)

| DTS range | Category | Meaning |
|-----------|----------|---------|
| < 0.3 | Shallow | Early retrieval / template completion |
| 0.3 – 0.7 | Moderate | Some policy or conditional reasoning |
| ≥ 0.7 | Deep | Multi-step inference, complex chains |

---

## Core classes (`dtr_analyzer.py`)

| Class | Purpose |
|-------|---------|
| `DTSConfig` | Hyperparameters: settling threshold `g`, depth fraction `ρ` |
| `TokenDTSMetrics` | Per-token: settling layer, JSD trace, deep-thinking flag |
| `PromptDTSAnalysis` | Aggregated DTS, SDF, average settling depth |
| `DeepThinkingAnalyzer` | Runs generation + layer-wise JSD analysis |

---

## Empirical results (LLaMA-3.1-8B-Instruct, NF4, Golden dataset)

| Split | n turns | Mean DTS |
|-------|---------|----------|
| PASS (Accuracy = 1) | 115 | 0.490 |
| FAIL (Accuracy = 0) | 56 | 0.592 |

FAIL conversations show ~10 pp higher DTS, consistent with harder queries
requiring deeper network computation.

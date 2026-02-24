"""
DTS Scoring Service — FastAPI Application
==========================================

Exposes a single scoring endpoint:

    POST /score
        Body: ScoreRequest  (prompt, optional generation / DTS config)
        Returns: ScoreResponse (dts_score, category, deep_tokens, all_tokens, …)

    GET  /health
        Returns: HealthResponse

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Interactive docs:  http://localhost:8000/docs
"""

import logging
import math
import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from dtr_analyzer import DTSConfig
from model_loader import get_analyzer, get_model_info, load_model
from schemas import HealthResponse, ScoreRequest, ScoreResponse, TokenDetail

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("dts_service")


# ── Lifespan: load model once at startup ─────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("DTS Scoring Service starting — loading model …")
    load_model()
    logger.info("Model ready.  Service is up.")
    yield
    logger.info("DTS Scoring Service shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="DTS Scoring Service",
    description=(
        "Score any prompt for reasoning depth using the **Deep Thinking Score (DTS)**.\n\n"
        "DTS measures the fraction of response tokens whose vocabulary prediction "
        "is resolved only in the deepest transformer layers — a proxy for how hard "
        "the model had to 'think' to answer."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper ────────────────────────────────────────────────────────────────────

def _category(dts: float) -> str:
    if dts < 0.3:
        return "Shallow"
    if dts < 0.7:
        return "Moderate"
    return "Deep"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Utility"])
def health() -> HealthResponse:
    """Returns service liveness and model-load status."""
    try:
        model_name, device, num_layers = get_model_info()
        loaded = True
    except Exception:
        model_name, device, num_layers = "", "", 0
        loaded = False

    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_name=model_name,
        device=device,
        num_layers=num_layers,
    )


@app.post("/score", response_model=ScoreResponse, tags=["Scoring"])
def score(request: ScoreRequest) -> ScoreResponse:
    """
    Generate a response for the given prompt using LLaMA-3.1-8B-Instruct,
    then compute the Deep Thinking Score (DTS) over all response tokens.

    **Returns**
    - `dts_score` — overall DTS ∈ [0, 1]
    - `category` — Shallow / Moderate / Deep
    - `deep_tokens` — list of tokens that required deep-layer settling
    - `all_tokens` — every response token with full per-token metrics
    """
    try:
        analyzer = get_analyzer()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Override per-request DTS hyperparameters if caller supplied them
    original_config = analyzer.config
    if (
        request.settling_threshold != original_config.settling_threshold
        or request.depth_fraction != original_config.depth_fraction
    ):
        analyzer.config = DTSConfig(
            settling_threshold=request.settling_threshold,
            depth_fraction=request.depth_fraction,
            device=original_config.device,
        )

    try:
        logger.info(
            "Scoring prompt (%d chars), max_new_tokens=%d",
            len(request.prompt),
            request.max_new_tokens,
        )
        analysis = analyzer.analyze_prompt(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
        )
    except Exception as exc:
        logger.exception("Error during DTS analysis")
        raise HTTPException(status_code=500, detail=f"DTS analysis failed: {exc}") from exc
    finally:
        # Restore original config
        analyzer.config = original_config

    model_name, _, num_layers = get_model_info()
    deep_layer_threshold = math.ceil(request.depth_fraction * num_layers)

    # ── Build per-token detail list ───────────────────────────────────────────
    all_token_details: list[TokenDetail] = []
    for pos, tm in enumerate(analysis.token_metrics):
        sdf = tm.settling_depth / num_layers if num_layers else 0.0
        detail = TokenDetail(
            position=pos,
            token=tm.token_text,
            token_id=tm.token_id,
            settling_layer=tm.settling_depth,
            settling_depth_fraction=round(sdf, 4),
            is_deep=tm.is_deep_thinking,
            jsd_trace=(
                [round(float(d), 4) for d in tm.distances]
                if request.include_jsd_trace
                else None
            ),
        )
        all_token_details.append(detail)

    deep_tokens = [t for t in all_token_details if t.is_deep]

    return ScoreResponse(
        dts_score=round(analysis.dts, 4),
        category=_category(analysis.dts),
        generated_response=analysis.generated_text,
        total_tokens=analysis.total_tokens,
        avg_settling_depth=round(float(analysis.avg_settling_depth), 4),
        settling_depth_fraction=round(float(analysis.settling_depth_fraction), 4),
        num_deep_tokens=len(deep_tokens),
        deep_tokens=deep_tokens,
        all_tokens=all_token_details,
        model_name=model_name,
        num_layers=num_layers,
        deep_layer_threshold=deep_layer_threshold,
        settling_threshold=request.settling_threshold,
        depth_fraction=request.depth_fraction,
    )

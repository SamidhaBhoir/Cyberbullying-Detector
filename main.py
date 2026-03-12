"""
main.py
-------
FastAPI backend for Cyberbullying Detection.

Endpoints:
  GET  /           → Welcome message
  GET  /health     → Server + model status
  POST /predict    → Single text prediction
  POST /predict/batch → Multiple texts at once

Run:
    uvicorn main:app --reload --port 8000

Docs:
    http://localhost:8000/docs
"""

import re
import os
import joblib
import numpy as np
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─── Paths ────────────────────────────────────────────────────────────────────
MODEL_PATH      = os.path.join(os.path.dirname(__file__), "model.pkl")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")

# ─── Global Model State ───────────────────────────────────────────────────────
model      = None
vectorizer = None

# ─── Keyword Blocklist (rule-based override layer) ────────────────────────────
# These words/phrases always flag as Bullying regardless of model output.
# Add or remove words here as needed.
BULLYING_KEYWORDS = [
    # Sexual / explicit
    "sex", "sexy", "sexual", "porn", "pornography", "nude", "naked",
    "boobs", "ass", "dick", "cock", "pussy", "vagina", "penis",
    "slut", "whore", "hoe", "prostitute", "rape", "molest",
    "horny", "masturbat", "orgasm", "erotic",
    # Hate / extreme slurs
    "nigger", "faggot", "retard", "kys", "kill yourself",
    "go die", "i will kill", "i will hurt",
]

# ─── Safe-phrase Whitelist ────────────────────────────────────────────────────
# If the text contains ANY of these positive words, the model alone cannot flag
# it as bullying (keyword blocklist can still override).
SAFE_WORDS = [
    "good", "great", "amazing", "awesome", "wonderful", "fantastic",
    "love", "lovely", "beautiful", "pretty", "cute", "kind", "sweet",
    "nice", "excellent", "brilliant", "perfect", "happy", "joy", "proud",
    "well done", "congrats", "congratulations", "thank", "thanks",
    "appreciate", "respect", "helpful", "friendly", "caring", "brave",
    "smart", "talented", "creative", "inspiring", "hope", "better", "omg",
]

def is_safe_phrase(text: str) -> bool:
    """Return True if the text contains clearly positive/safe words."""
    lower = text.lower()
    for word in SAFE_WORDS:
        if re.search(r"\b" + re.escape(word), lower):
            return True
    return False

def check_keywords(text: str) -> list[str]:
    """Return list of matched blocklist keywords found in the text."""
    lower = text.lower()
    matched = []
    for kw in BULLYING_KEYWORDS:
        pattern = r"\b" + re.escape(kw)
        if re.search(pattern, lower):
            matched.append(kw)
    return matched

# ─── Text Cleaning (must match train_model.py) ────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ─── Lifespan (load model on startup) ────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, vectorizer
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model file not found at '{MODEL_PATH}'. "
            "Please run 'python train_model.py' first."
        )
    if not os.path.exists(VECTORIZER_PATH):
        raise RuntimeError(
            f"Vectorizer file not found at '{VECTORIZER_PATH}'. "
            "Please run 'python train_model.py' first."
        )
    print("[*] Loading model and vectorizer...")
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("[OK] Model loaded successfully!")
    yield
    # Cleanup (if needed)
    model      = None
    vectorizer = None

# ─── App Initialization ───────────────────────────────────────────────────────
app = FastAPI(
    title="🛡️ Cyberbullying Detection API",
    description=(
        "Detects whether a comment, phrase, or sentence is **cyberbullying** or not. "
        "Trained on the 'Suspicious Communication on Social Platforms' dataset."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow all origins (for frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Pydantic Schemas ─────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        example="You are so stupid and ugly, nobody likes you.",
    )

class PredictResponse(BaseModel):
    text: str
    cleaned_text: str
    label: str                      # "Bullying" or "Not Bullying"
    is_bullying: bool
    confidence: float               # probability of the predicted class
    bullying_probability: float
    not_bullying_probability: float
    flagged_by: str                 # "model", "keyword_blocklist", or "both"
    triggered_keywords: List[str]   # keywords that matched (empty if model-only)

class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        example=[
            "You are so stupid and ugly!",
            "Have a great day, hope you feel better soon.",
        ],
    )

class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    total: int
    bullying_count: int
    not_bullying_count: int

# ─── Helper: run prediction ───────────────────────────────────────────────────
def run_prediction(text: str) -> PredictResponse:
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Try again shortly.")

    cleaned = clean_text(text)
    if not cleaned:
        raise HTTPException(status_code=422, detail="Text is empty after cleaning.")

    # 1️⃣  ML model prediction
    vec   = vectorizer.transform([cleaned])
    proba = model.predict_proba(vec)[0]   # [P(not bullying), P(bullying)]
    not_bully_prob  = float(proba[0])
    bully_prob      = float(proba[1])
    # Raised threshold to 0.60 to reduce false positives on borderline phrases.
    # Also override to safe if text contains clearly positive words.
    model_bullying  = bully_prob >= 0.60 and not is_safe_phrase(text)

    # 2️⃣  Keyword blocklist check
    matched_keywords = check_keywords(text)
    keyword_bullying = len(matched_keywords) > 0

    # 3️⃣  Combine: either source can flag as bullying
    is_bullying = model_bullying or keyword_bullying

    # Determine who flagged it
    if model_bullying and keyword_bullying:
        flagged_by = "both"
    elif keyword_bullying:
        flagged_by = "keyword_blocklist"
        # Override probabilities to reflect the forced flag
        bully_prob     = max(bully_prob, 0.95)
        not_bully_prob = 1.0 - bully_prob
    else:
        flagged_by = "model"

    label      = "Bullying" if is_bullying else "Not Bullying"
    confidence = bully_prob if is_bullying else not_bully_prob

    return PredictResponse(
        text=text,
        cleaned_text=cleaned,
        label=label,
        is_bullying=is_bullying,
        confidence=round(confidence, 4),
        bullying_probability=round(bully_prob, 4),
        not_bullying_probability=round(not_bully_prob, 4),
        flagged_by=flagged_by,
        triggered_keywords=matched_keywords,
    )

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", tags=["General"])
def root():
    return {
        "message": "🛡️ Cyberbullying Detection API is running!",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
        "batch_predict": "POST /predict/batch",
    }


@app.get("/health", tags=["General"])
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
    }


@app.post("/predict", response_model=PredictResponse, tags=["Detection"])
def predict(request: PredictRequest):
    """
    Analyze a **single** comment or phrase and detect if it is cyberbullying.

    - **text**: The comment/phrase to analyze (1–5000 characters)

    Returns the label (`Bullying` / `Not Bullying`), confidence score,
    and individual class probabilities.
    """
    return run_prediction(request.text)


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Detection"])
def predict_batch(request: BatchPredictRequest):
    """
    Analyze **multiple** comments or phrases in a single request (up to 100).

    Returns individual predictions for each text plus a summary count.
    """
    results = [run_prediction(text) for text in request.texts]
    bullying_count = sum(1 for r in results if r.is_bullying)
    return BatchPredictResponse(
        results=results,
        total=len(results),
        bullying_count=bullying_count,
        not_bullying_count=len(results) - bullying_count,
    )

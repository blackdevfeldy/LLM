#!/usr/bin/env python3
"""
SENTRA ML - API FastAPI
API REST pour la détection de fraudes SMS
"""

import os
import sys
import re
import time
import uuid
import pickle
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Literal

from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

# Config — fonctionne à la fois en local et dans Docker
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

try:
    from config import (
        MODELS_DIR, DISTILBERT_DIR, RF_MODEL_PATH, TFIDF_PATH, MAX_LENGTH,
        API_HOST, API_PORT, API_KEY, ALLOWED_ORIGINS, REDIS_URL, ENVIRONMENT,
    )
except ImportError:
    # Fallback Docker : les modèles sont dans /app/models
    MODELS_DIR = Path(os.getenv("MODELS_DIR", "/app/models"))
    DISTILBERT_DIR = MODELS_DIR / "distilbert_sms_final"
    RF_MODEL_PATH = MODELS_DIR / "sentra_random_forest_model.pkl"
    TFIDF_PATH = MODELS_DIR / "sentra_tfidf_vectorizer.pkl"
    MAX_LENGTH = 128
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    API_KEY = os.getenv("SENTRA_API_KEY", "")
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",") if os.getenv("ALLOWED_ORIGINS") else []
    REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
    ENVIRONMENT = os.getenv("ENVIRONMENT", "production")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Prometheus metrics ────────────────────────────────────────────────────────
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

PROM_REQUESTS = Counter("sentra_requests_total", "Total API requests", ["endpoint", "status"])
PROM_LATENCY = Histogram("sentra_predict_latency_seconds", "Prediction latency")
PROM_MODEL_LOADED = Gauge("sentra_model_loaded", "1 if model is loaded")
PROM_UPTIME = Gauge("sentra_uptime_seconds", "API uptime")

_start_time = time.time()

# ── Redis (optionnel — dégradation gracieuse) ─────────────────────────────────
_redis = None
try:
    import redis as _redis_lib
    _redis = _redis_lib.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=2)
    _redis.ping()
    logger.info(f"✓ Redis connecté ({REDIS_URL})")
except Exception:
    _redis = None
    logger.warning("Redis indisponible — rate limiting désactivé")

RATE_LIMIT_WINDOW = 60   # secondes
RATE_LIMIT_MAX = 60       # requêtes / fenêtre

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="SENTRA ML API",
    description="API de détection de fraudes SMS pour le projet SENTRA BENIN",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["*"] if ENVIRONMENT == "development" else ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# ── Sécurité : API Key ───────────────────────────────────────────────────────
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: Optional[str] = Depends(_api_key_header)):
    if not API_KEY:
        return None  # pas de clé configurée → accès libre (dev)
    if api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Clé API invalide")
    return api_key

# ── Rate limiting middleware ──────────────────────────────────────────────────
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if _redis and request.url.path == "/predict":
        client_ip = request.client.host
        key = f"rate:{client_ip}"
        try:
            current = _redis.incr(key)
            if current == 1:
                _redis.expire(key, RATE_LIMIT_WINDOW)
            if current > RATE_LIMIT_MAX:
                PROM_REQUESTS.labels(endpoint="/predict", status="429").inc()
                return PlainTextResponse("Rate limit dépassé", status_code=429)
        except Exception:
            pass  # dégradation gracieuse
    return await call_next(request)

# ── Pydantic models ──────────────────────────────────────────────────────────
class SMSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Texte du SMS à analyser")
    user_id: Optional[str] = Field(None, description="ID utilisateur optionnel")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "URGENT: Votre compte M-Pesa a été suspendu. Appelez le +22912345678",
                "user_id": "user_12345",
            }
        }

class SMSResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    legitimate_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: Literal["low", "medium", "high"]
    model_used: str
    processing_time_ms: float
    timestamp: str
    request_id: str

class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: bool
    redis_connected: bool
    timestamp: str

# ── Modèle ML (Ensemble BERT + RF) ───────────────────────────────────────────
class SENTRAModel:
    """Wrapper API autour de l'ensemble predictor (BERT + RF weighted voting)."""

    def __init__(self):
        self.model = None
        self.model_name = "none"
        self._load()

    def _load(self):
        # Essai 1 : Ensemble (BERT + RF)
        try:
            import numpy as np
            self._np = np
            self._load_ensemble()
            return
        except Exception as e:
            logger.warning(f"Ensemble non disponible, fallback single model : {e}")

        # Essai 2 : DistilBERT seul
        if not self._try_load_transformer():
            # Essai 3 : Random Forest seul
            self._load_rf_fallback()

        PROM_MODEL_LOADED.set(1 if self.model is not None else 0)

    def _load_ensemble(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import numpy as np

        # Charger les deux modèles
        self._bert_tokenizer = AutoTokenizer.from_pretrained(str(DISTILBERT_DIR))
        self._bert_model = AutoModelForSequenceClassification.from_pretrained(str(DISTILBERT_DIR))
        self._bert_model.eval()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._bert_model.to(self._device)

        with open(RF_MODEL_PATH, "rb") as f:
            self._rf_model = pickle.load(f)
        with open(TFIDF_PATH, "rb") as f:
            self._tfidf = pickle.load(f)

        self._bert_weight = 0.65
        self._rf_weight = 0.35
        self.model = True  # flag : modèle disponible
        self.model_name = "ensemble(bert+rf)"
        PROM_MODEL_LOADED.set(1)
        logger.info(f"✓ Ensemble chargé (BERT={self._bert_weight}, RF={self._rf_weight}) sur {self._device}")

    def _try_load_transformer(self) -> bool:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            if not DISTILBERT_DIR.exists():
                return False

            self._bert_tokenizer = AutoTokenizer.from_pretrained(str(DISTILBERT_DIR))
            self._bert_model = AutoModelForSequenceClassification.from_pretrained(str(DISTILBERT_DIR))
            self._bert_model.eval()
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._bert_model.to(self._device)
            self.model = True
            self.model_name = "distilbert"
            PROM_MODEL_LOADED.set(1)
            logger.info(f"✓ DistilBERT chargé sur {self._device}")
            return True
        except Exception as e:
            logger.error(f"Erreur chargement DistilBERT : {e}")
            return False

    def _load_rf_fallback(self):
        try:
            with open(RF_MODEL_PATH, "rb") as f:
                self._rf_model = pickle.load(f)
            with open(TFIDF_PATH, "rb") as f:
                self._tfidf = pickle.load(f)
            self.model = True
            self.model_name = "random_forest"
            PROM_MODEL_LOADED.set(1)
            logger.info("✓ Random Forest (fallback) chargé")
        except Exception as e:
            logger.error(f"Erreur chargement RF : {e}")
            self.model = None
            PROM_MODEL_LOADED.set(0)

    # ── Prédiction ────────────────────────────────────────────────────────
    def predict(self, text: str) -> dict:
        if self.model is None:
            raise HTTPException(status_code=503, detail="Modèle non disponible")

        t0 = time.time()

        if self.model_name.startswith("ensemble"):
            fraud_prob, legit_prob = self._predict_ensemble(text)
        elif self.model_name == "distilbert":
            fraud_prob, legit_prob = self._predict_bert(text)
        else:
            fraud_prob, legit_prob = self._predict_rf(text)

        latency_ms = (time.time() - t0) * 1000

        if fraud_prob < 0.3:
            risk = "low"
        elif fraud_prob < 0.7:
            risk = "medium"
        else:
            risk = "high"

        return {
            "is_fraud": fraud_prob >= 0.5,
            "fraud_probability": float(fraud_prob),
            "legitimate_probability": float(legit_prob),
            "risk_level": risk,
            "processing_time_ms": latency_ms,
            "model_used": self.model_name,
        }

    def _predict_bert(self, text: str):
        import torch
        inputs = self._bert_tokenizer(
            text, padding="max_length", truncation=True,
            max_length=MAX_LENGTH, return_tensors="pt",
        ).to(self._device)
        with torch.no_grad():
            probs = torch.softmax(self._bert_model(**inputs).logits, dim=-1)
        return probs[0][1].item(), probs[0][0].item()

    def _predict_rf(self, text: str):
        cleaned = self._clean(text)
        features = self._tfidf.transform([cleaned])
        probs = self._rf_model.predict_proba(features)[0]
        return float(probs[1]), float(probs[0])

    def _predict_ensemble(self, text: str):
        import numpy as np
        bert_fraud, bert_legit = self._predict_bert(text)
        rf_fraud, rf_legit = self._predict_rf(text)

        fraud_prob = self._bert_weight * bert_fraud + self._rf_weight * rf_fraud
        legit_prob = self._bert_weight * bert_legit + self._rf_weight * rf_legit
        return fraud_prob, legit_prob

    @staticmethod
    def _clean(text: str) -> str:
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\+?\d{1,3}[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{2,4}[-.\s]?\d{2,4}', '', text)
        text = re.sub(r'[^a-zA-ZÀ-ÿ0-9\s.!?]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

model = SENTRAModel()

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "SENTRA ML API", "version": "3.0.0", "docs": "/docs"}

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy" if model.model is not None else "degraded",
        version="3.0.0",
        models_loaded=model.model is not None,
        redis_connected=_redis is not None,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

@app.post("/predict", response_model=SMSResponse)
async def predict_endpoint(
    request: SMSRequest,
    _: Optional[str] = Depends(verify_api_key),
):
    """Prédit si un SMS est une fraude (1-2000 caractères)."""
    try:
        result = model.predict(request.text)
        PROM_LATENCY.observe(result["processing_time_ms"] / 1000)
        PROM_REQUESTS.labels(endpoint="/predict", status="200").inc()

        return SMSResponse(
            **result,
            timestamp=datetime.now(timezone.utc).isoformat(),
            request_id=uuid.uuid4().hex[:12],
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Erreur prédiction")
        PROM_REQUESTS.labels(endpoint="/predict", status="500").inc()
        raise HTTPException(status_code=500, detail="Erreur interne du serveur")

@app.get("/metrics")
async def metrics_endpoint():
    """Métriques au format Prometheus."""
    PROM_UPTIME.set(time.time() - _start_time)
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)

#!/usr/bin/env python3
"""
SENTRA ML - Ensemble Predictor
Combine DistilBERT + Random Forest via weighted voting pour une meilleure détection.
"""

import sys
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import logging

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    MODELS_DIR, DISTILBERT_DIR, RF_MODEL_PATH, TFIDF_PATH, MAX_LENGTH,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Poids par défaut : DistilBERT a plus de poids car meilleur F1
BERT_WEIGHT = 0.65
RF_WEIGHT = 0.35


class EnsemblePredictor:
    """Combine DistilBERT + Random Forest par weighted voting"""

    def __init__(self, bert_weight: float = BERT_WEIGHT, rf_weight: float = RF_WEIGHT):
        self.bert_weight = bert_weight
        self.rf_weight = rf_weight
        self.bert_loaded = False
        self.rf_loaded = False

        # Charger Random Forest
        try:
            with open(RF_MODEL_PATH, 'rb') as f:
                self.rf_model = pickle.load(f)
            with open(TFIDF_PATH, 'rb') as f:
                self.tfidf = pickle.load(f)
            self.rf_loaded = True
            logger.info("✓ Random Forest chargé")
        except Exception as e:
            logger.warning(f"⚠ Random Forest non disponible: {e}")

        # Charger DistilBERT
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(DISTILBERT_DIR))
            self.bert_model = AutoModelForSequenceClassification.from_pretrained(str(DISTILBERT_DIR))
            self.bert_model.eval()
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.bert_model.to(self.device)
            self.bert_loaded = True
            logger.info("✓ DistilBERT chargé")
        except Exception as e:
            logger.warning(f"⚠ DistilBERT non disponible: {e}")

        if not self.bert_loaded and not self.rf_loaded:
            raise RuntimeError("Aucun modèle disponible")

        logger.info(f"✓ Ensemble prêt (BERT={self.bert_weight}, RF={self.rf_weight})")

    def _predict_bert(self, text: str) -> np.ndarray:
        """Retourne [p_legit, p_fraud] depuis DistilBERT"""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=MAX_LENGTH, padding="max_length"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.bert_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        return probs

    def _predict_rf(self, text: str) -> np.ndarray:
        """Retourne [p_legit, p_fraud] depuis Random Forest"""
        features = self.tfidf.transform([text])
        probs = self.rf_model.predict_proba(features)[0]
        return probs

    def predict(self, text: str) -> Dict:
        """
        Prédiction ensemble par weighted voting sur les probabilités.
        """
        if not text or not text.strip():
            return {
                'is_fraud': False, 'fraud_probability': 0.0,
                'risk_level': 'unknown', 'method': 'none'
            }

        probs_list = []
        weights = []
        methods = []

        if self.bert_loaded:
            bert_probs = self._predict_bert(text)
            probs_list.append(bert_probs)
            weights.append(self.bert_weight)
            methods.append('bert')

        if self.rf_loaded:
            rf_probs = self._predict_rf(text)
            probs_list.append(rf_probs)
            weights.append(self.rf_weight)
            methods.append('rf')

        # Normaliser les poids
        total_w = sum(weights)
        weights = [w / total_w for w in weights]

        # Weighted average des probabilités
        ensemble_probs = np.zeros(2)
        for prob, w in zip(probs_list, weights):
            ensemble_probs += w * prob

        fraud_prob = float(ensemble_probs[1])

        if fraud_prob < 0.3:
            risk_level = 'low'
        elif fraud_prob < 0.7:
            risk_level = 'medium'
        else:
            risk_level = 'high'

        return {
            'is_fraud': fraud_prob >= 0.5,
            'fraud_probability': fraud_prob,
            'legitimate_probability': float(ensemble_probs[0]),
            'risk_level': risk_level,
            'method': '+'.join(methods),
        }

    def predict_batch(self, texts: list) -> list:
        return [self.predict(t) for t in texts]


def main():
    """Démonstration de l'ensemble predictor"""
    print("=" * 60)
    print("🚀 SENTRA ML - Ensemble Predictor (BERT + RF)")
    print("=" * 60)

    predictor = EnsemblePredictor()

    test_messages = [
        "Bonjour, comment ça va? On se voit demain?",
        "URGENT: Votre compte M-Pesa a été suspendu. Appelez immédiatement le +22912345678",
        "Félicitations! Vous avez gagné 1000000 FCFA. Envoyez CODE au 12345",
        "Ok je serai là à 14h. Bisous",
        "ALERTE: Tentative de connexion suspecte. Cliquez ici: http://bit.ly/abc123",
        "Maman m'a dit de te dire qu'elle arrive ce soir",
        "Votre transfert de 50000 CFA a échoué. Contactez le service client au 229-XXX",
    ]

    print("\n📱 Tests de prédiction:\n")
    for i, msg in enumerate(test_messages, 1):
        result = predictor.predict(msg)
        status = "🚨 FRAUDE" if result['is_fraud'] else "✅ LÉGITIME"
        print(f"{i}. {status} [{result['method']}]")
        print(f"   \"{msg[:65]}{'...' if len(msg) > 65 else ''}\"")
        print(f"   P(fraude)={result['fraud_probability']:.2%}  Risque={result['risk_level'].upper()}")
        print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
SENTRA ML - Script d'Inférence
Prédiction en temps réel pour les SMS
"""

import sys
import pickle
import re
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODELS_DIR

class SENTRAPredictor:
    """Classe pour la prédiction de fraude SMS"""
    
    def __init__(self, model_dir: str = None):
        """Initialise le prédicteur avec les modèles sauvegardés"""
        self.model_dir = Path(model_dir) if model_dir else MODELS_DIR
        
        # Charger le modèle
        model_path = self.model_dir / "sentra_random_forest_model.pkl"
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Charger le vectorizer TF-IDF
        tfidf_path = self.model_dir / "sentra_tfidf_vectorizer.pkl"
        with open(tfidf_path, 'rb') as f:
            self.tfidf = pickle.load(f)
        
        # Patterns de nettoyage
        self.url_pattern = re.compile(r'http\S+|www\S+|https\S+', flags=re.MULTILINE)
        self.phone_pattern = re.compile(r'\+?\d{1,3}[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{2,4}[-.\s]?\d{2,4}')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.special_chars_pattern = re.compile(r'[^a-zA-ZÀ-ÿ0-9\s\.\!\?]')
        self.extra_spaces_pattern = re.compile(r'\s+')
        
        print("✅ Modèle SENTRA chargé (Random Forest)")
    
    def clean_text(self, text: str) -> str:
        """Nettoie le texte du SMS"""
        if not isinstance(text, str):
            return ""
        
        # Conversion en minuscules
        text = text.lower()
        
        # Suppression des URLs
        text = self.url_pattern.sub('', text)
        
        # Suppression des emails
        text = self.email_pattern.sub('', text)
        
        # Suppression des numéros de téléphone
        text = self.phone_pattern.sub('', text)
        
        # Suppression des caractères spéciaux
        text = self.special_chars_pattern.sub(' ', text)
        
        # Suppression des espaces multiples
        text = self.extra_spaces_pattern.sub(' ', text)
        
        return text.strip()
    
    def predict(self, text: str) -> Dict:
        """
        Prédit si un SMS est une fraude
        
        Returns:
            Dict avec is_fraud, fraud_probability, risk_level
        """
        # Nettoyer le texte
        cleaned_text = self.clean_text(text)
        
        if not cleaned_text:
            return {
                'is_fraud': False,
                'fraud_probability': 0.0,
                'legitimate_probability': 1.0,
                'risk_level': 'unknown',
                'error': 'Texte vide après nettoyage'
            }
        
        # Vectoriser
        features = self.tfidf.transform([cleaned_text])
        
        # Prédire
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        fraud_prob = float(probabilities[1])
        legit_prob = float(probabilities[0])
        
        # Déterminer le niveau de risque
        if fraud_prob < 0.3:
            risk_level = 'low'
        elif fraud_prob < 0.7:
            risk_level = 'medium'
        else:
            risk_level = 'high'
        
        return {
            'is_fraud': bool(prediction),
            'fraud_probability': fraud_prob,
            'legitimate_probability': legit_prob,
            'risk_level': risk_level,
            'cleaned_text': cleaned_text
        }
    
    def predict_batch(self, texts: list) -> list:
        """Prédit sur une liste de SMS"""
        return [self.predict(text) for text in texts]


def main():
    """Fonction principale pour démonstration"""
    print("=" * 60)
    print("🚀 SENTRA ML - Test d'Inférence")
    print("=" * 60)
    
    # Initialiser le prédicteur
    predictor = SENTRAPredictor()
    
    # Exemples de SMS à tester
    test_messages = [
        "Bonjour, comment ça va? On se voit demain?",
        "URGENT: Votre compte M-Pesa a été suspendu. Appelez immédiatement le +22912345678",
        "Félicitations! Vous avez gagné 1000000 FCFA. Envoyez CODE au 12345 pour réclamer",
        "Transfert de 5000 FCFA reçu de Jean. Votre solde est de 15000 FCFA",
        "ALERTE: Tentative de connexion suspecte détectée. Cliquez ici: http://bit.ly/abc123"
    ]
    
    print("\n📱 Tests de prédiction:\n")
    
    for i, message in enumerate(test_messages, 1):
        result = predictor.predict(message)
        
        status = "🚨 FRAUDE" if result['is_fraud'] else "✅ LÉGITIME"
        
        print(f"{i}. {status}")
        print(f"   Message: {message[:60]}...")
        print(f"   Probabilité fraude: {result['fraud_probability']:.2%}")
        print(f"   Niveau de risque: {result['risk_level'].upper()}")
        print()
    
    print("=" * 60)


if __name__ == "__main__":
    main()

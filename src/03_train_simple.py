#!/usr/bin/env python3
"""
SENTRA ML - Script d'Entraînement Simplifié (Random Forest uniquement)
Entraînement avec scikit-learn uniquement
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
from typing import Dict, Tuple

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)
from imblearn.over_sampling import SMOTE

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    PROCESSED_DIR, MODELS_DIR, RANDOM_STATE, METRICS_PATH,
    RF_MODEL_PATH, TFIDF_PATH, TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE,
    TFIDF_MIN_DF, TFIDF_MAX_DF, RF_N_ESTIMATORS, RF_MAX_DEPTH, TEST_SIZE,
)

def load_data() -> pd.DataFrame:
    """Charge les données prétraitées"""
    print("Chargement des données...")
    
    data_path = PROCESSED_DIR / "processed_data.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Données non trouvées: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"✓ {len(df)} instances chargées")
    print(f"  Fraudes: {(df['label'] == 1).sum()}")
    print(f"  Légitimes: {(df['label'] == 0).sum()}")
    
    return df

def prepare_features(df: pd.DataFrame) -> Tuple:
    """Prépare les features TF-IDF"""
    print("\nPréparation des features TF-IDF...")
    
    # Nettoyer les NaN dans les labels et textes
    df = df.dropna(subset=['label', 'text_cleaned'])
    df['label'] = df['label'].astype(int)
    
    texts = df['text_cleaned'].astype(str)
    
    # TF-IDF
    tfidf = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
    )
    
    X = tfidf.fit_transform(texts)
    y = df['label'].values
    
    print(f"✓ Shape: {X.shape}")
    
    return X, y, tfidf

def split_data(X, y) -> Tuple:
    """Divise les données"""
    print("\nDivision train/test...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"✓ Train: {len(y_train)} (fraudes: {sum(y_train)})") 
    print(f"✓ Test: {len(y_test)}")
    
    # SMOTE sur le train set uniquement
    print("\nApplication SMOTE (oversampling)...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"✓ Train après SMOTE: {len(y_train)} (fraudes: {sum(y_train)})")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Entraîne Random Forest"""
    print("\nEntraînement Random Forest...")
    
    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight='balanced',
    )
    
    model.fit(X_train, y_train)
    print("✓ Modèle entraîné")
    
    return model

def evaluate_model(model, X_test, y_test) -> Dict:
    """Évalue le modèle"""
    print("\nÉvaluation...")
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    }
    
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1']:.4f}")
    
    return metrics

def cross_validate(model, X_train, y_train) -> Dict:
    """Validation croisée sur le train set uniquement (pas de data leakage)"""
    print("\nValidation croisée (5-fold sur train)...")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
    
    print(f"  F1 scores: {scores}")
    print(f"  F1 moyen: {scores.mean():.4f}")
    
    return {'cv_f1_mean': scores.mean(), 'cv_f1_std': scores.std()}

def save_model(model, tfidf, metrics):
    """Sauvegarde le modèle"""
    print("\nSauvegarde...")
    
    with open(RF_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    with open(TFIDF_PATH, 'wb') as f:
        pickle.dump(tfidf, f)
    
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Modèle: {RF_MODEL_PATH}")
    print(f"✓ Vectorizer: {TFIDF_PATH}")
    print(f"✓ Métriques: {METRICS_PATH}")

def main():
    print("=" * 60)
    print("🚀 SENTRA ML - Entraînement Random Forest")
    print("=" * 60)
    
    try:
        # Charger
        df = load_data()
        
        # Features
        X, y, tfidf = prepare_features(df)
        
        # Split
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Entraîner
        model = train_model(X_train, y_train)
        
        # Évaluer
        metrics = evaluate_model(model, X_test, y_test)
        
        # CV (sur train uniquement pour éviter data leakage)
        cv_results = cross_validate(model, X_train, y_train)
        metrics.update(cv_results)
        
        # Sauvegarder
        save_model(model, tfidf, metrics)
        
        print("\n" + "=" * 60)
        print("✅ Entraînement terminé!")
        print("=" * 60)
        print(f"\n🏆 Résultats:")
        print(f"   F1-Score: {metrics['f1']:.4f}")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   CV F1: {metrics['cv_f1_mean']:.4f}")
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

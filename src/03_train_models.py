#!/usr/bin/env python3
"""
SENTRA ML - Script d'Entraînement des Modèles
Entraînement Random Forest et XGBoost avec validation croisée
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import logging
from typing import Dict, Tuple

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            classification_report, confusion_matrix, roc_auc_score)
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DIR, MODELS_DIR, RANDOM_STATE, TEST_SIZE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VAL_SIZE = 0.1


class SENTRAModelTrainer:
    """Classe pour l'entraînement des modèles SENTRA"""
    
    def __init__(self):
        self.tfidf = None
        self.rf_model = None
        self.xgb_model = None
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Charge les données prétraitées"""
        logger.info("Chargement des données...")
        
        # Charger les données textuelles
        data_path = PROCESSED_DIR / "processed_data.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Données non trouvées: {data_path}. Exécutez d'abord 02_preprocessing.py")
        
        df = pd.read_csv(data_path)
        
        # Charger les features manuelles
        features_path = PROCESSED_DIR / "manual_features.csv"
        if features_path.exists():
            features_df = pd.read_csv(features_path)
        else:
            features_df = pd.DataFrame()
        
        logger.info(f"✓ {len(df)} instances chargées")
        
        return df, features_df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple:
        """Prépare les features TF-IDF et manuelles"""
        logger.info("\n🔧 Préparation des features...")
        
        # TF-IDF
        self.tfidf = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            stop_words='english'
        )
        
        # Combiner les features TF-IDF
        X_tfidf = self.tfidf.fit_transform(df['text_cleaned'].fillna(''))
        self.feature_names = self.tfidf.get_feature_names_out().tolist()
        
        # Features manuelles
        manual_feature_cols = [c for c in df.columns if c.startswith(('keyword_', 'has_', 'fraud_', 'length', 'word', 'avg', 'digit', 'uppercase', 'exclamation', 'question', 'dollar', 'percent', 'uppercase_ratio'))]
        X_manual = df[manual_feature_cols].values if manual_feature_cols else np.zeros((len(df), 1))
        
        # Combiner
        from scipy.sparse import hstack, csr_matrix
        X_combined = hstack([X_tfidf, csr_matrix(X_manual)])
        
        y = df['label'].values
        
        logger.info(f"✓ Features TF-IDF: {X_tfidf.shape[1]}")
        logger.info(f"✓ Features manuelles: {X_manual.shape[1]}")
        logger.info(f"✓ Total features: {X_combined.shape[1]}")
        
        return X_combined, y
    
    def split_data(self, X, y) -> Tuple:
        """Divise les données en train/val/test"""
        logger.info("\n📊 Division des données...")
        
        # Split train / temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=TEST_SIZE + VAL_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        
        # Split temp en val / test
        val_ratio = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
        )
        
        logger.info(f"✓ Train: {len(y_train)} ({len(y_train)/len(y)*100:.1f}%)")
        logger.info(f"✓ Validation: {len(y_val)} ({len(y_val)/len(y)*100:.1f}%)")
        logger.info(f"✓ Test: {len(y_test)} ({len(y_test)/len(y)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_random_forest(self, X_train, y_train) -> RandomForestClassifier:
        """Entraîne un modèle Random Forest"""
        logger.info("\n🌲 Entraînement Random Forest...")
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        
        logger.info("✓ Random Forest entraîné")
        
        return model
    
    def train_xgboost(self, X_train, y_train) -> xgb.XGBClassifier:
        """Entraîne un modèle XGBoost"""
        logger.info("\n⚡ Entraînement XGBoost...")
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric='logloss'
        )
        
        model.fit(X_train, y_train)
        
        logger.info("✓ XGBoost entraîné")
        
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name: str) -> Dict:
        """Évalue un modèle sur le set de test"""
        logger.info(f"\n📈 Évaluation {model_name}...")
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1']:.4f}")
        logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def cross_validate(self, model, X_train, y_train, model_name: str) -> Dict:
        """Validation croisée 5-fold sur train uniquement (pas de data leakage)"""
        logger.info(f"\n🔍 Validation croisée {model_name} (sur train)...")
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
        
        logger.info(f"  F1 scores: {scores}")
        logger.info(f"  F1 moyen: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return {'cv_f1_mean': scores.mean(), 'cv_f1_std': scores.std()}
    
    def select_best_model(self, rf_metrics: Dict, xgb_metrics: Dict) -> str:
        """Sélectionne le meilleur modèle basé sur le F1-score"""
        logger.info("\n🏆 Sélection du meilleur modèle...")
        
        if xgb_metrics['f1'] > rf_metrics['f1']:
            self.best_model = self.xgb_model
            self.best_model_name = 'xgboost'
            logger.info(f"✓ XGBoost sélectionné (F1: {xgb_metrics['f1']:.4f})")
        else:
            self.best_model = self.rf_model
            self.best_model_name = 'random_forest'
            logger.info(f"✓ Random Forest sélectionné (F1: {rf_metrics['f1']:.4f})")
        
        return self.best_model_name
    
    def save_models(self):
        """Sauvegarde tous les modèles et les métadonnées"""
        logger.info("\n💾 Sauvegarde des modèles...")
        
        # Sauvegarder le meilleur modèle
        best_model_path = MODELS_DIR / f"sentra_{self.best_model_name}_model.pkl"
        joblib.dump(self.best_model, best_model_path)
        logger.info(f"✓ Meilleur modèle: {best_model_path}")
        
        # Sauvegarder Random Forest
        rf_path = MODELS_DIR / "sentra_random_forest_model.pkl"
        joblib.dump(self.rf_model, rf_path)
        
        # Sauvegarder XGBoost
        xgb_path = MODELS_DIR / "sentra_xgboost_model.pkl"
        joblib.dump(self.xgb_model, xgb_path)
        
        # Sauvegarder le vectorizer TF-IDF
        tfidf_path = MODELS_DIR / "sentra_tfidf_vectorizer.pkl"
        joblib.dump(self.tfidf, tfidf_path)
        logger.info(f"✓ Vectorizer TF-IDF: {tfidf_path}")
        
        # Sauvegarder les features names
        feature_names_path = MODELS_DIR / "sentra_feature_names.json"
        with open(feature_names_path, 'w') as f:
            json.dump(self.feature_names, f)
        
        logger.info("✓ Tous les modèles sauvegardés")
    
    def run(self):
        """Exécute le pipeline complet d'entraînement"""
        logger.info("=" * 60)
        logger.info("🚀 SENTRA ML - Entraînement des Modèles")
        logger.info("=" * 60)
        
        # Charger les données
        df, features_df = self.load_data()
        
        # Préparer les features
        X, y = self.prepare_features(df)
        
        # Diviser les données
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Entraîner Random Forest
        self.rf_model = self.train_random_forest(X_train, y_train)
        rf_metrics = self.evaluate_model(self.rf_model, X_test, y_test, "Random Forest")
        rf_cv = self.cross_validate(self.rf_model, X_train, y_train, "Random Forest")
        rf_metrics.update(rf_cv)
        
        # Entraîner XGBoost
        self.xgb_model = self.train_xgboost(X_train, y_train)
        xgb_metrics = self.evaluate_model(self.xgb_model, X_test, y_test, "XGBoost")
        xgb_cv = self.cross_validate(self.xgb_model, X_train, y_train, "XGBoost")
        xgb_metrics.update(xgb_cv)
        
        # Sélectionner le meilleur modèle
        best_name = self.select_best_model(rf_metrics, xgb_metrics)
        
        # Sauvegarder
        self.save_models()
        
        # Sauvegarder les métriques
        metrics = {
            'random_forest': rf_metrics,
            'xgboost': xgb_metrics,
            'best_model': best_name,
            'training_samples': len(y_train),
            'test_samples': len(y_test),
            'feature_count': X.shape[1]
        }
        
        metrics_path = MODELS_DIR / "sentra_model_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Résumé final
        logger.info("\n" + "=" * 60)
        logger.info("✅ Entraînement terminé!")
        logger.info("=" * 60)
        logger.info(f"\n🏆 Meilleur modèle: {best_name.upper()}")
        logger.info(f"   F1-Score: {max(rf_metrics['f1'], xgb_metrics['f1']):.4f}")
        logger.info(f"   Précision: {max(rf_metrics['precision'], xgb_metrics['precision']):.4f}")
        logger.info(f"   Recall: {max(rf_metrics['recall'], xgb_metrics['recall']):.4f}")
        
        return metrics


def main():
    """Fonction principale"""
    trainer = SENTRAModelTrainer()
    metrics = trainer.run()
    return metrics


if __name__ == "__main__":
    main()

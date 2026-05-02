#!/usr/bin/env python3
"""
SENTRA ML - Pipeline de Prétraitement des Données
Nettoyage, normalisation et préparation des datasets
"""

import sys
import pandas as pd
import numpy as np
import re
import string
from pathlib import Path
from typing import List, Dict
import logging

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_DIR, PROCESSED_DIR, FRAUD_KEYWORDS as CFG_FRAUD_KEYWORDS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Patterns regex pour le nettoyage
URL_PATTERN = re.compile(r'http\S+|www\S+|https\S+', flags=re.MULTILINE)
PHONE_PATTERN = re.compile(r'\+?\d{1,3}[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{2,4}[-.\s]?\d{2,4}')
EMAIL_PATTERN = re.compile(r'\S+@\S+')
SPECIAL_CHARS_PATTERN = re.compile(r'[^a-zA-ZÀ-ÿ0-9\s\.\!\?]')
EXTRA_SPACES_PATTERN = re.compile(r'\s+')
CURRENCY_PATTERN = re.compile(r'[£$€]\s?\d+[,.]?\d*|\d+[,.]?\d*\s?(?:usd|eur|gbp|fcfa|cfa|xof)', re.IGNORECASE)
REPEATED_CHARS_PATTERN = re.compile(r'(.)\1{2,}')

# Abréviations SMS courantes (EN + FR) → expansion pour meilleure tokenization
SMS_ABBREVIATIONS = {
    'u': 'you', 'ur': 'your', 'r': 'are', 'y': 'why', 'b': 'be',
    'c': 'see', 'k': 'ok', 'n': 'and', 'pls': 'please', 'plz': 'please',
    'msg': 'message', 'msgs': 'messages', 'txt': 'text', 'txts': 'texts',
    'thx': 'thanks', 'thnx': 'thanks', 'tnx': 'thanks',
    'asap': 'as soon as possible', 'btw': 'by the way',
    'fyi': 'for your information', 'imo': 'in my opinion',
    'omg': 'oh my god', 'lol': 'laughing', 'brb': 'be right back',
    'idk': 'i do not know', 'tbh': 'to be honest',
    'w/': 'with', 'w/o': 'without', 'b4': 'before', '2day': 'today',
    '2nite': 'tonight', '4u': 'for you', '2u': 'to you',
    'acc': 'account', 'acct': 'account', 'amt': 'amount',
    'govt': 'government', 'recv': 'receive', 'rcv': 'receive',
    'svc': 'service', 'info': 'information', 'ref': 'reference',
    'no.': 'number', 'num': 'number', 'tel': 'telephone',
    'xtra': 'extra', 'mins': 'minutes', 'secs': 'seconds',
    'smth': 'something', 'sms': 'message',
    # FR
    'slt': 'salut', 'bjr': 'bonjour', 'bsr': 'bonsoir',
    'stp': 's il te plait', 'svp': 's il vous plait',
    'mrc': 'merci', 'bcp': 'beaucoup', 'pk': 'pourquoi',
    'pcq': 'parce que', 'ns': 'nous', 'vs': 'vous',
    'qd': 'quand', 'ds': 'dans', 'pr': 'pour', 'tt': 'tout',
    'tjs': 'toujours', 'pb': 'probleme', 'cb': 'combien',
    'cpte': 'compte', 'tel': 'telephone', 'num': 'numero',
    'rdv': 'rendez vous', 'auj': 'aujourd hui',
}

FRAUD_KEYWORDS = CFG_FRAUD_KEYWORDS

class TextPreprocessor:
    """Classe pour le prétraitement des textes SMS"""
    
    def __init__(self):
        self.fraud_keywords = [kw.lower() for kw in FRAUD_KEYWORDS]
    
    def expand_abbreviations(self, text: str) -> str:
        """
        Expanse les abréviations SMS pour améliorer la tokenization
        """
        words = text.split()
        expanded = []
        for word in words:
            clean_word = word.strip('.,!?')
            if clean_word in SMS_ABBREVIATIONS:
                expanded.append(SMS_ABBREVIATIONS[clean_word])
            else:
                expanded.append(word)
        return ' '.join(expanded)
    
    def normalize_repeated_chars(self, text: str) -> str:
        """
        Réduit les caractères répétés (ex: 'freeee' → 'free')
        """
        return REPEATED_CHARS_PATTERN.sub(r'\1\1', text)
    
    def replace_currency(self, text: str) -> str:
        """
        Remplace les montants par un token générique (signal de fraude)
        """
        return CURRENCY_PATTERN.sub(' MONEY_AMOUNT ', text)
    
    def clean_text(self, text: str) -> str:
        """
        Nettoie et normalise le texte
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Conversion en minuscules
        text = text.lower()
        
        # Remplacement des montants par token générique
        text = self.replace_currency(text)
        
        # Suppression des URLs
        text = URL_PATTERN.sub('', text)
        
        # Suppression des emails
        text = EMAIL_PATTERN.sub('', text)
        
        # Suppression des numéros de téléphone (mais on garde l'info comme feature)
        text = PHONE_PATTERN.sub('', text)
        
        # Expansion des abréviations SMS
        text = self.expand_abbreviations(text)
        
        # Normalisation des caractères répétés
        text = self.normalize_repeated_chars(text)
        
        # Suppression des caractères spéciaux sauf ponctuation basique
        text = SPECIAL_CHARS_PATTERN.sub(' ', text)
        
        # Suppression des espaces multiples
        text = EXTRA_SPACES_PATTERN.sub(' ', text)
        
        # Strip final
        text = text.strip()
        
        return text
    
    def extract_features(self, text: str) -> Dict:
        """
        Extrait des features linguistiques du texte
        """
        if not text:
            return self._empty_features()
        
        features = {}
        text_lower = text.lower()
        
        # Features de base
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        # Caractères spéciaux
        features['digit_count'] = sum(c.isdigit() for c in text)
        features['uppercase_count'] = sum(c.isupper() for c in text)
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['dollar_count'] = text.count('$')
        features['percent_count'] = text.count('%')
        
        # Présence d'éléments suspects
        features['has_url'] = int(bool(URL_PATTERN.search(text)))
        features['has_phone'] = int(bool(PHONE_PATTERN.search(text)))
        features['has_email'] = int(bool(EMAIL_PATTERN.search(text)))
        features['has_currency'] = int(bool(CURRENCY_PATTERN.search(text)))
        
        # Compteur d'abréviations SMS (signal de spam/phishing)
        words_lower = text_lower.split()
        features['abbreviation_count'] = sum(1 for w in words_lower if w.strip('.,!?') in SMS_ABBREVIATIONS)
        
        # Ratio de chiffres (montants, numéros = signal)
        features['digit_ratio'] = features['digit_count'] / max(len(text), 1)
        
        # Mots de fraude
        for keyword in self.fraud_keywords:
            features[f'keyword_{keyword.replace(" ", "_")}'] = int(keyword in text_lower)
        
        # Score de suspicion global
        fraud_score = sum([features[f'keyword_{kw.replace(" ", "_")}'] for kw in self.fraud_keywords])
        features['fraud_keyword_score'] = fraud_score
        
        # Ratio majuscules
        if len(text) > 0:
            features['uppercase_ratio'] = features['uppercase_count'] / len(text)
        else:
            features['uppercase_ratio'] = 0
        
        return features
    
    def _empty_features(self) -> Dict:
        """Retourne des features vides pour un texte vide"""
        features = {
            'length': 0,
            'word_count': 0,
            'avg_word_length': 0,
            'digit_count': 0,
            'uppercase_count': 0,
            'exclamation_count': 0,
            'question_count': 0,
            'dollar_count': 0,
            'percent_count': 0,
            'has_url': 0,
            'has_phone': 0,
            'has_email': 0,
            'has_currency': 0,
            'abbreviation_count': 0,
            'digit_ratio': 0,
            'fraud_keyword_score': 0,
            'uppercase_ratio': 0
        }
        
        for keyword in self.fraud_keywords:
            features[f'keyword_{keyword.replace(" ", "_")}'] = 0
        
        return features


def load_and_merge_datasets() -> pd.DataFrame:
    """
    Charge et fusionne tous les datasets disponibles
    """
    datasets = []
    
    # 1. MOZ-Smishing (utiliser le fichier nettoyé)
    moz_path = RAW_DIR / "moz_smishing_clean.csv"
    if moz_path.exists():
        logger.info("Chargement MOZ-Smishing (clean)...")
        df_moz = pd.read_csv(moz_path)
        
        # Standardiser les colonnes - le fichier clean a déjà les labels 0/1
        if 'text' in df_moz.columns and 'label' in df_moz.columns:
            # Supprimer les lignes avec NaN
            df_moz = df_moz.dropna(subset=['label', 'text'])
            df_moz['label'] = df_moz['label'].astype(int)
            df_moz = df_moz[['text', 'label']].copy()
            df_moz['source'] = 'moz_smishing'
            df_moz['language'] = 'portuguese'
            datasets.append(df_moz)
            logger.info(f"  ✓ {len(df_moz)} instances chargées")
    
    # 2. Phishing Dataset
    phishing_path = RAW_DIR / "phishing_dataset.csv"
    if phishing_path.exists():
        logger.info("Chargement Phishing Dataset...")
        df_phish = pd.read_csv(phishing_path)
        
        # Standardiser les colonnes
        if 'text' in df_phish.columns and 'label' in df_phish.columns:
            df_phish = df_phish[['text', 'label']].copy()
            df_phish['source'] = 'phishing_dataset'
            df_phish['language'] = 'mixed'
            datasets.append(df_phish)
            logger.info(f"  ✓ {len(df_phish)} instances chargées")
    
    # 3. UCI SMS Spam
    uci_path = RAW_DIR / "uci_sms_spam.csv"
    if uci_path.exists():
        logger.info("Chargement UCI SMS...")
        df_uci = pd.read_csv(uci_path)
        
        # Standardiser les labels (ham=0, spam=1 si texte, sinon garder entiers)
        if df_uci['label'].dtype == object:
            df_uci['label'] = df_uci['label'].map({'ham': 0, 'spam': 1})
        df_uci['label'] = df_uci['label'].astype(int)
        df_uci = df_uci.dropna(subset=['label', 'text'])
        df_uci = df_uci[['text', 'label']].copy()
        df_uci['source'] = 'uci_sms'
        df_uci['language'] = 'english'
        datasets.append(df_uci)
        logger.info(f"  ✓ {len(df_uci)} instances chargées")
    
    if not datasets:
        raise ValueError("Aucun dataset trouvé. Veuillez d'abord exécuter 01_download_datasets.py")
    
    # Fusionner
    df_merged = pd.concat(datasets, ignore_index=True)
    logger.info(f"\n📊 Total fusionné: {len(df_merged)} instances")
    logger.info(f"Distribution des labels: {df_merged['label'].value_counts().to_dict()}")
    
    return df_merged


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique le prétraitement complet sur le DataFrame
    """
    preprocessor = TextPreprocessor()
    
    logger.info("\n🔧 Prétraitement des textes...")
    
    # Nettoyage des textes
    df['text_cleaned'] = df['text'].apply(preprocessor.clean_text)
    
    # Supprimer les textes vides après nettoyage
    df = df[df['text_cleaned'].str.len() > 0].reset_index(drop=True)
    logger.info(f"  ✓ {len(df)} instances après suppression des textes vides")
    
    # Extraction des features
    logger.info("🔍 Extraction des features...")
    features_list = df['text'].apply(preprocessor.extract_features).tolist()
    features_df = pd.DataFrame(features_list)
    
    # Concaténer avec le DataFrame principal
    df = pd.concat([df, features_df], axis=1)
    
    logger.info(f"  ✓ {len(features_df.columns)} features extraites")
    
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les doublons basés sur le texte nettoyé
    """
    initial_count = len(df)
    df = df.drop_duplicates(subset=['text_cleaned'], keep='first')
    final_count = len(df)
    
    removed = initial_count - final_count
    logger.info(f"\n🗑️  Doublons supprimés: {removed}")
    logger.info(f"   Restant: {final_count} instances")
    
    return df


def save_processed_data(df: pd.DataFrame):
    """
    Sauvegarde les données prétraitées
    """
    # Sauvegarder le dataset complet
    output_path = PROCESSED_DIR / "processed_data.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"\n💾 Données sauvegardées: {output_path}")
    
    # Sauvegarder uniquement les features
    feature_cols = [col for col in df.columns if col.startswith(('keyword_', 'has_', 'fraud_')) or col in 
                    ['length', 'word_count', 'avg_word_length', 'digit_count', 'uppercase_count',
                     'exclamation_count', 'question_count', 'dollar_count', 'percent_count', 'uppercase_ratio']]
    
    features_df = df[feature_cols + ['label']].copy()
    features_path = PROCESSED_DIR / "manual_features.csv"
    features_df.to_csv(features_path, index=False)
    logger.info(f"💾 Features manuelles sauvegardées: {features_path}")
    
    # Valider les labels avant stats
    valid_mask = df['label'].isin([0, 1])
    if not valid_mask.all():
        invalid_count = (~valid_mask).sum()
        logger.warning(f"  ⚠️ {invalid_count} instances avec labels invalides supprimées")
        df = df[valid_mask].reset_index(drop=True)

    fraud_count = int((df['label'] == 1).sum())
    legit_count = int((df['label'] == 0).sum())
    assert fraud_count + legit_count == len(df), "Incohérence labels: fraud + legit != total"

    stats = {
        'total_instances': len(df),
        'fraud_instances': fraud_count,
        'legitimate_instances': legit_count,
        'fraud_ratio': float(fraud_count / len(df)) if len(df) > 0 else 0.0,
        'sources': df['source'].value_counts().to_dict()
    }
    
    import json
    stats_path = PROCESSED_DIR / "statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"💾 Statistiques sauvegardées: {stats_path}")


def main():
    """Fonction principale"""
    logger.info("=" * 60)
    logger.info("🔧 SENTRA ML - Pipeline de Prétraitement")
    logger.info("=" * 60)
    
    try:
        # Charger les datasets
        df = load_and_merge_datasets()
        
        # Prétraitement
        df = preprocess_dataframe(df)
        
        # Supprimer les doublons
        df = remove_duplicates(df)
        
        # Sauvegarder
        save_processed_data(df)
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ Prétraitement terminé avec succès!")
        logger.info("=" * 60)
        
        # Résumé final
        logger.info(f"\n📊 RÉSUMÉ:")
        logger.info(f"   Total instances: {len(df)}")
        logger.info(f"   Fraudes: {(df['label'] == 1).sum()}")
        logger.info(f"   Légitimes: {(df['label'] == 0).sum()}")
        # Compter les features
        feature_count = len([c for c in df.columns if c.startswith(('keyword_', 'has_', 'fraud_', 'length', 'word', 'avg', 'digit', 'uppercase', 'exclamation', 'question', 'dollar', 'percent', 'uppercase_ratio'))])
        logger.info(f"   Features: {feature_count}")
        
    except Exception as e:
        logger.error(f"\n❌ Erreur: {e}")
        raise


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
SENTRA ML - Script de téléchargement des datasets
Télécharge les 3 datasets principaux pour la Phase 2
"""

import os
import sys
import pandas as pd
from datasets import load_dataset
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_DIR

DATA_DIR = RAW_DIR

def download_moz_smishing():
    """Télécharge MOZ-Smishing depuis HuggingFace"""
    print("📥 Téléchargement MOZ-Smishing...")
    
    try:
        dataset = load_dataset("MOZNLP/MOZ-Smishing")
        
        # Vérifier les splits disponibles
        print(f"   Splits disponibles: {list(dataset.keys())}")
        
        # Utiliser le premier split disponible
        split_name = list(dataset.keys())[0]
        df = pd.DataFrame(dataset[split_name])
        
        # Sauvegarder
        output_path = DATA_DIR / "moz_smishing.csv"
        df.to_csv(output_path, index=False)
        
        print(f"✅ MOZ-Smishing sauvegardé: {output_path}")
        print(f"   Taille: {len(df)} instances")
        print(f"   Colonnes: {list(df.columns)}")
        print(f"   Split utilisé: {split_name}")
        
        return df
        
    except Exception as e:
        print(f"❌ Erreur MOZ-Smishing: {e}")
        return None

def download_phishing_dataset():
    """Télécharge le dataset phishing depuis HuggingFace (format Parquet)"""
    print("\n📥 Téléchargement Phishing Dataset...")

    # Liste de datasets compatibles (format Parquet, sans loading script)
    candidates = [
        {
            "id": "shawhin/phishing-site-classification",
            "text_col": "text",
            "label_col": "labels",
            "label_map": None,  # déjà 0/1
        },
    ]

    for candidate in candidates:
        try:
            print(f"   Essai: {candidate['id']}")
            dataset = load_dataset(candidate["id"])

            split_name = list(dataset.keys())[0]
            df = pd.DataFrame(dataset[split_name])

            # Normaliser les colonnes text / label
            text_col = candidate["text_col"]
            label_col = candidate["label_col"]

            if text_col not in df.columns or label_col not in df.columns:
                print(f"   ⚠️  Colonnes attendues ({text_col}, {label_col}) absentes. "
                      f"Colonnes trouvées: {list(df.columns)}")
                continue

            df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})

            if candidate["label_map"]:
                df["label"] = df["label"].map(candidate["label_map"])

            output_path = DATA_DIR / "phishing_dataset.csv"
            df.to_csv(output_path, index=False)

            print(f"✅ Phishing Dataset sauvegardé: {output_path}")
            print(f"   Source: {candidate['id']}")
            print(f"   Taille: {len(df)} instances")
            print(f"   Distribution: {df['label'].value_counts().to_dict()}")
            print(f"   Split utilisé: {split_name}")

            return df

        except Exception as e:
            print(f"   ⚠️  Échec {candidate['id']}: {e}")
            continue

    print("❌ Tous les datasets phishing ont échoué.")
    return None

def download_uci_sms():
    """Télécharge UCI SMS Spam Collection depuis l'URL directe"""
    print("\n📥 Téléchargement UCI SMS Spam Collection...")
    
    try:
        # URL alternative - fichier texte direct
        url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
        output_path = DATA_DIR / "uci_sms_spam.tsv"
        
        urllib.request.urlretrieve(url, output_path)
        
        # Lire et convertir
        df = pd.read_csv(output_path, sep='\t', header=None, names=['label', 'text'])
        
        # Sauvegarder aussi en CSV
        csv_path = DATA_DIR / "uci_sms_spam.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"✅ UCI SMS sauvegardé: {csv_path}")
        print(f"   Taille: {len(df)} instances")
        print(f"   Distribution: {df['label'].value_counts().to_dict()}")
        
        return df
        
    except Exception as e:
        print(f"❌ Erreur UCI SMS: {e}")
        # Créer un fichier vide si échec
        return None

def main():
    """Fonction principale"""
    print("=" * 60)
    print("🚀 SENTRA ML - Téléchargement des Datasets")
    print("=" * 60)
    
    results = {}
    
    # Télécharger les datasets
    results['moz'] = download_moz_smishing()
    results['phishing'] = download_phishing_dataset()
    results['uci'] = download_uci_sms()
    
    # Résumé
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ")
    print("=" * 60)
    
    total = 0
    for name, df in results.items():
        if df is not None:
            count = len(df)
            total += count
            print(f"✅ {name}: {count} instances")
        else:
            print(f"❌ {name}: ÉCHEC")
    
    print(f"\n📈 Total: {total} instances téléchargées")
    print(f"📁 Dossier: {DATA_DIR}")
    
    return results

if __name__ == "__main__":
    main()

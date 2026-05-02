#!/usr/bin/env python3
"""
SENTRA ML - Nettoyage des données MOZ-Smishing
Corrige les problèmes de formatage CSV
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_DIR

def clean_moz_data():
    """Nettoie le dataset MOZ-Smishing"""
    print("Nettoyage MOZ-Smishing...")
    
    # Lire avec pandas (gère mieux les multilignes)
    df = pd.read_csv(RAW_DIR / "moz_smishing.csv")
    
    print(f"Colonnes: {df.columns.tolist()}")
    print(f"Taille initiale: {len(df)}")
    print(f"\nValeurs uniques de label:")
    print(df['label'].value_counts())
    
    # Mapper les labels
    label_map = {'Legitimate': 0, 'Smishing': 1}
    df['label'] = df['label'].map(label_map)
    
    # Supprimer les NaN
    df_clean = df.dropna(subset=['label', 'text'])
    
    print(f"\nTaille après nettoyage: {len(df_clean)}")
    print(f"Fraudes: {(df_clean['label'] == 1).sum()}")
    print(f"Légitimes: {(df_clean['label'] == 0).sum()}")
    
    # Sauvegarder
    output_path = RAW_DIR / "moz_smishing_clean.csv"
    df_clean.to_csv(output_path, index=False)
    print(f"\nSauvegardé: {output_path}")
    
    return df_clean

if __name__ == "__main__":
    clean_moz_data()

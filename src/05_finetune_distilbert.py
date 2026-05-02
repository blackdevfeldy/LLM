#!/usr/bin/env python3
"""
SENTRA ML - Script de Fine-tuning DistilBERT
Fine-tuning sur les datasets MOZ-Smishing et UCI SMS
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    PROCESSED_DIR, MODELS_DIR, LOGS_DIR, DISTILBERT_DIR, DISTILBERT_CHECKPOINT_DIR,
    DISTILBERT_MODEL, DISTILBERT_BATCH_SIZE, DISTILBERT_LEARNING_RATE,
    DISTILBERT_EPOCHS, DISTILBERT_WARMUP_STEPS, DISTILBERT_WEIGHT_DECAY,
    MAX_LENGTH, RANDOM_STATE, TEST_SIZE,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_MODEL = DISTILBERT_MODEL
BATCH_SIZE = DISTILBERT_BATCH_SIZE
LEARNING_RATE = DISTILBERT_LEARNING_RATE
NUM_EPOCHS = DISTILBERT_EPOCHS
WARMUP_STEPS = DISTILBERT_WARMUP_STEPS
WEIGHT_DECAY = DISTILBERT_WEIGHT_DECAY


def load_processed_data() -> pd.DataFrame:
    """Charge les données prétraitées"""
    logger.info("Chargement des données prétraitées...")
    
    data_path = PROCESSED_DIR / "processed_data.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Données non trouvées: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Nettoyer les NaN
    df = df.dropna(subset=['text_cleaned', 'label'])
    df['label'] = df['label'].astype(int)
    
    logger.info(f"✓ {len(df)} instances chargées")
    logger.info(f"  Fraudes: {(df['label'] == 1).sum()}")
    logger.info(f"  Légitimes: {(df['label'] == 0).sum()}")
    
    return df


def prepare_dataset(df: pd.DataFrame) -> DatasetDict:
    """Prépare le dataset au format HuggingFace"""
    logger.info("\nPréparation du dataset HuggingFace...")
    
    # Renommer les colonnes pour HuggingFace
    df = df[['text_cleaned', 'label']].copy()
    df.columns = ['text', 'label']
    
    # Split train/val/test
    train_df, temp_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df['label']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=RANDOM_STATE, stratify=temp_df['label']
    )
    
    logger.info(f"✓ Train: {len(train_df)}")
    logger.info(f"✓ Validation: {len(val_df)}")
    logger.info(f"✓ Test: {len(test_df)}")
    
    # Créer DatasetDict
    dataset = DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'validation': Dataset.from_pandas(val_df),
        'test': Dataset.from_pandas(test_df)
    })
    
    return dataset


def tokenize_dataset(dataset: DatasetDict, tokenizer) -> DatasetDict:
    """Tokenize le dataset"""
    logger.info("\nTokenization...")
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    
    logger.info("✓ Tokenization terminée")
    return tokenized_dataset


def compute_metrics(pred) -> Dict:
    """Calcule les métriques d'évaluation"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


class WeightedTrainer(Trainer):
    """Trainer avec weighted cross-entropy loss pour gérer le déséquilibre de classes"""
    
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fn = nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fn = nn.CrossEntropyLoss()
        
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def train_model(tokenized_dataset: DatasetDict, model, tokenizer, class_weights=None) -> Trainer:
    """Entraîne le modèle avec weighted loss"""
    logger.info("\nConfiguration de l'entraînement...")
    
    training_args = TrainingArguments(
        output_dir=str(DISTILBERT_CHECKPOINT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        learning_rate=LEARNING_RATE,
        logging_dir=str(LOGS_DIR),
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        seed=RANDOM_STATE,
        fp16=torch.cuda.is_available(),
    )
    
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    if class_weights is not None:
        logger.info(f"✓ Weighted loss activé: {class_weights}")
    logger.info("✓ Début de l'entraînement...")
    trainer.train()
    
    return trainer


def evaluate_model(trainer, tokenized_dataset):
    """Évalue le modèle sur le test set"""
    logger.info("\nÉvaluation sur le test set...")
    
    results = trainer.evaluate(tokenized_dataset['test'])
    
    logger.info("\n📊 Résultats sur Test Set:")
    for key, value in results.items():
        logger.info(f"  {key}: {value:.4f}")
    
    return results


def save_model(trainer, tokenizer):
    """Sauvegarde le modèle et le tokenizer"""
    logger.info("\nSauvegarde du modèle...")
    
    # Sauvegarder avec le trainer
    trainer.save_model(DISTILBERT_DIR)
    tokenizer.save_pretrained(DISTILBERT_DIR)
    
    logger.info(f"✓ Modèle sauvegardé: {DISTILBERT_DIR}")


def main():
    """Fonction principale"""
    logger.info("=" * 60)
    logger.info("🚀 SENTRA ML - Fine-tuning DistilBERT")
    logger.info("=" * 60)
    
    try:
        # Vérifier device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"\nDevice: {device}")
        if device == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Charger les données
        df = load_processed_data()
        
        # Préparer le dataset
        dataset = prepare_dataset(df)
        
        # Charger tokenizer et modèle
        logger.info(f"\nChargement du modèle: {BASE_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL,
            num_labels=2,
            id2label={0: "LEGITIMATE", 1: "FRAUD"},
            label2id={"LEGITIMATE": 0, "FRAUD": 1}
        )
        
        # Tokenizer
        tokenized_dataset = tokenize_dataset(dataset, tokenizer)
        
        # Calculer les poids de classes pour la weighted loss
        train_labels = np.array(dataset['train']['label'])
        weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_labels)
        logger.info(f"\nPoids de classes: legit={weights[0]:.3f}, fraud={weights[1]:.3f}")
        
        # Entraîner
        trainer = train_model(tokenized_dataset, model, tokenizer, class_weights=weights)
        
        # Évaluer
        results = evaluate_model(trainer, tokenized_dataset)
        
        # Sauvegarder
        save_model(trainer, tokenizer)
        
        # Résumé final
        logger.info("\n" + "=" * 60)
        logger.info("✅ Fine-tuning terminé!")
        logger.info("=" * 60)
        logger.info(f"\n🏆 F1-Score: {results.get('eval_f1', 0):.4f}")
        logger.info(f"🏆 Accuracy: {results.get('eval_accuracy', 0):.4f}")
        logger.info(f"🏆 Precision: {results.get('eval_precision', 0):.4f}")
        logger.info(f"🏆 Recall: {results.get('eval_recall', 0):.4f}")
        
    except Exception as e:
        logger.error(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

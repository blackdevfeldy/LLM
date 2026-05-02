#!/usr/bin/env python3
"""
SENTRA ML - Script de Conversion TensorFlow Lite
Convertit le modèle DistilBERT PyTorch vers TensorFlow Lite avec quantization INT8
"""

import sys
import torch
import tensorflow as tf
from pathlib import Path
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODELS_DIR, DISTILBERT_DIR, MAX_LENGTH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_pytorch_model():
    """Charge le modèle PyTorch fine-tuné"""
    logger.info("Chargement du modèle PyTorch...")
    
    if not DISTILBERT_DIR.exists():
        raise FileNotFoundError(f"Modèle non trouvé: {DISTILBERT_DIR}")
    
    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_DIR)
    model.eval()
    
    logger.info("✓ Modèle PyTorch chargé")
    return model, tokenizer

def create_representative_dataset(tokenizer, num_samples=100):
    """Crée un dataset représentatif pour la calibration INT8"""
    logger.info("Création du dataset représentatif...")
    
    # Exemples de SMS pour calibration
    sample_texts = [
        "URGENT: Votre compte a été suspendu",
        "Félicitations! Vous avez gagné",
        "Transfert reçu de",
        "Bonjour, comment ça va?",
        "Code de vérification",
        "Remboursement approuvé",
        "Rendez-vous demain",
        "ALERTE: Tentative de connexion",
        "Votre solde est de",
        "Cliquez ici pour réclamer"
    ] * 10  # 100 samples
    
    def representative_dataset():
        for text in sample_texts[:num_samples]:
            inputs = tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors='pt'
            )
            # Convertir en format TF Lite
            input_ids = inputs['input_ids'].numpy().astype(np.int32)
            attention_mask = inputs['attention_mask'].numpy().astype(np.int32)
            yield [input_ids, attention_mask]
    
    return representative_dataset

def convert_to_tflite(model, tokenizer):
    """Convertit le modèle vers TensorFlow Lite avec quantization INT8"""
    logger.info("\nConversion vers TensorFlow Lite...")
    
    # Sauvegarder temporairement au format SavedModel
    temp_dir = MODELS_DIR / "temp_tf_model"
    
    try:
        # Exporter vers ONNX puis TensorFlow
        logger.info("Export vers ONNX...")
        dummy_input = torch.zeros(1, MAX_LENGTH, dtype=torch.long)
        dummy_mask = torch.ones(1, MAX_LENGTH, dtype=torch.long)
        
        torch.onnx.export(
            model,
            (dummy_input, dummy_mask),
            str(MODELS_DIR / "distilbert_sms.onnx"),
            input_names=['input_ids', 'attention_mask'],
            output_names=['output'],
            dynamic_axes={
                'input_ids': {0: 'batch_size'},
                'attention_mask': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            opset_version=11
        )
        logger.info("✓ Export ONNX terminé")
        
        # Convertir ONNX vers TensorFlow
        logger.info("Conversion ONNX vers TensorFlow...")
        import onnx
        from onnx_tf.backend import prepare
        
        onnx_model = onnx.load(str(MODELS_DIR / "distilbert_sms.onnx"))
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(str(temp_dir))
        logger.info("✓ Conversion TF terminée")
        
        # Convertir vers TensorFlow Lite avec quantization INT8
        logger.info("Conversion TFLite avec quantization INT8...")
        converter = tf.lite.TFLiteConverter.from_saved_model(str(temp_dir))
        
        # Configuration quantization INT8
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        # Dataset représentatif pour calibration
        converter.representative_dataset = create_representative_dataset(tokenizer)
        
        # Convertir
        tflite_model = converter.convert()
        
        # Sauvegarder
        tflite_path = MODELS_DIR / "sentra_distilbert_int8.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        # Stats
        size_mb = len(tflite_model) / (1024 * 1024)
        logger.info(f"✓ Modèle TFLite sauvegardé: {tflite_path}")
        logger.info(f"  Taille: {size_mb:.2f} MB")
        
        # Comparer avec modèle original
        pytorch_size = (MODELS_DIR / "distilbert_sms_final" / "pytorch_model.bin").stat().st_size / (1024 * 1024)
        compression = pytorch_size / size_mb
        logger.info(f"  Compression: {compression:.1f}x (PyTorch: {pytorch_size:.1f} MB)")
        
        return tflite_path
        
    except Exception as e:
        logger.error(f"❌ Erreur conversion: {e}")
        # Fallback: quantization dynamique sans dataset représentatif
        logger.info("Tentative avec quantization dynamique...")
        converter = tf.lite.TFLiteConverter.from_saved_model(str(temp_dir))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        tflite_path = MODELS_DIR / "sentra_distilbert_dynamic.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        logger.info(f"✓ Modèle TFLite (dynamic) sauvegardé: {tflite_path}")
        return tflite_path
    
    finally:
        # Cleanup
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        if (MODELS_DIR / "distilbert_sms.onnx").exists():
            (MODELS_DIR / "distilbert_sms.onnx").unlink()

def test_tflite_inference(tflite_path, tokenizer):
    """Test l'inférence avec le modèle TFLite"""
    logger.info("\nTest inférence TFLite...")
    
    # Charger modèle TFLite
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    # Obtenir détails des tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    logger.info(f"  Input shape: {input_details[0]['shape']}")
    logger.info(f"  Output shape: {output_details[0]['shape']}")
    
    # Test
    test_texts = [
        "URGENT: Votre compte a été suspendu. Appelez immédiatement!",
        "Bonjour, on se voit demain pour le déjeuner?"
    ]
    
    for text in test_texts:
        # Tokenizer
        inputs = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors='np'
        )
        
        # Préparer input
        input_ids = inputs['input_ids'].astype(np.int32)
        attention_mask = inputs['attention_mask'].astype(np.int32)
        
        # Inférence
        interpreter.set_tensor(input_details[0]['index'], input_ids)
        if len(input_details) > 1:
            interpreter.set_tensor(input_details[1]['index'], attention_mask)
        
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        # Résultat
        pred_class = np.argmax(output, axis=-1)[0]
        prob = tf.nn.softmax(output[0]).numpy()
        
        status = "FRAUDE" if pred_class == 1 else "LÉGITIME"
        logger.info(f"  \"{text[:40]}...\" → {status} ({prob[pred_class]:.2%})")

def main():
    """Fonction principale"""
    logger.info("=" * 60)
    logger.info("🚀 SENTRA ML - Conversion TensorFlow Lite")
    logger.info("=" * 60)
    
    try:
        # Vérifier dépendances
        try:
            import onnx
            import onnx_tf
        except ImportError:
            logger.warning("Installation des dépendances ONNX...")
            import subprocess
            subprocess.run([
                "conda", "run", "-n", "base", "pip", "install", 
                "onnx", "onnx-tf", "-q"
            ], check=True)
        
        # Charger modèle
        model, tokenizer = load_pytorch_model()
        
        # Convertir
        tflite_path = convert_to_tflite(model, tokenizer)
        
        # Tester
        test_tflite_inference(tflite_path, tokenizer)
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ Conversion terminée!")
        logger.info("=" * 60)
        logger.info(f"\n📱 Modèle mobile prêt: {tflite_path}")
        
    except Exception as e:
        logger.error(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

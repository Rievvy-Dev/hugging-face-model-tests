# -*- coding: utf-8 -*-
"""Configurações do pipeline de fine-tuning."""
import os
import torch

# ==================== DISPOSITIVO ====================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[CONFIG] Device: {device}")

# ==================== MODELOS ====================
MODELS = {
    "helsinki": "Helsinki-NLP/opus-mt-tc-big-en-pt",
    "m2m100": "danhsf/m2m100_418M-finetuned-kde4-en-to-pt_BR",
}

# ==================== DADOS SCIELO ====================
SCIELO_ABSTRACTS_FILE = "abstracts_scielo.csv"  # Arquivo completo (2.7M)
SCIELO_TRAIN_CSV = "finetuning/abstracts-datasets/scielo_abstracts_train.csv"  # Treino
SCIELO_VAL_CSV = "finetuning/abstracts-datasets/scielo_abstracts_val.csv"      # Validação (interno)
SCIELO_TEST_CSV = "finetuning/abstracts-datasets/scielo_abstracts_test.csv"    # Teste (never seen)

# Padrões de divisão padrão
DEFAULT_TRAIN_SAMPLES = 80_000
DEFAULT_VAL_SAMPLES = 20_000
DEFAULT_TEST_SAMPLES = 20_000

SEED = 42  # Seed para reprodutibilidade

# ==================== RESUMOS ====================
TRAIN_RESUME_DIR = "./checkpoints/training"  # Checkpoints de treino
EVAL_RESUME_DIR = "./checkpoints/evaluation"  # Checkpoints de avaliação

# Criar diretórios se não existirem
os.makedirs(TRAIN_RESUME_DIR, exist_ok=True)
os.makedirs(EVAL_RESUME_DIR, exist_ok=True)

# ==================== FINE-TUNING ====================
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 8
DEFAULT_EVAL_BATCH_SIZE = 8
DEFAULT_LR = 2e-5
DEFAULT_WARMUP_STEPS = 500
DEFAULT_MAX_SEQ_LEN = 256

# ==================== SAÍDA ====================
OUTPUT_DIR = "./"
BEFORE_METRICS_FILE = "scielo_before_finetuning.csv"
AFTER_METRICS_FILE = "scielo_after_finetuning.csv"
COMPARISON_REPORT = "SCIENCE_EVALUATION_REPORT.txt"

# ==================== MÉTRICAS ====================
METRICS = ["bleu", "chrf", "comet", "bertscore"]

# Thresholds de alerta (overfitting)
BLEU_OVERFITTING_THRESHOLD = 20.0  # BLEU increase > 20%
BLEU_DEGRADATION_THRESHOLD = -10.0  # BLEU decrease > 10%

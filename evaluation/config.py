# -*- coding: utf-8 -*-
"""Constantes e configuração do pipeline de avaliação."""
import os
import torch

# Dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"

# Datasets
USE_FULL_DATASETS = True
PARACRAWL_N = 5000
MAX_SAMPLES_WMT24PP = 5000
MAX_SAMPLES_PARACRAWL = 5_000
MAX_SAMPLES_FLORES = 1012
MAX_SAMPLES_OPUS100 = 5_000

# Modelos
MODEL_NAMES = [
    "Helsinki-NLP/opus-mt-tc-big-en-pt",
    "Narrativa/mbart-large-50-finetuned-opus-en-pt-translation",
    "unicamp-dl/translation-en-pt-t5",
    "VanessaSchenkel/unicamp-finetuned-en-to-pt-dataset-ted",
    "danhsf/m2m100_418M-finetuned-kde4-en-to-pt_BR",
    "aimped/nlp-health-translation-base-en-pt",
]

DANHSF_MODEL = "danhsf/m2m100_418M-finetuned-kde4-en-to-pt_BR"

# Saída: pasta e nomes de arquivos
OUTPUT_DIR = "evaluation_results"
OUTPUT_FILE_ALL = "translation_metrics_all.csv"


def get_output_dir():
    """Diretório absoluto dos resultados (relativo à raiz do projeto)."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, OUTPUT_DIR)


def get_batch_sizes():
    """Retorna dict (dataset_name, model_name) -> batch_size."""
    models = list(MODEL_NAMES)
    datasets = ["wmt24pp", "paracrawl", "flores", "opus100"]
    base = {(ds, m): 1 for ds in datasets for m in models}
    for ds in ("paracrawl", "opus100"):
        base[(ds, DANHSF_MODEL)] = 8
    return base


def get_csv_header():
    """Cabeçalho do CSV de resultados."""
    return [
        "Dataset", "Tamanho Real", "Modelo", "Device", "Batch Size", "Tamanho Dataset (usado)",
        "Exemplos do Dataset", "Total de Sentenças", "Total de Palavras", "Média Palavras por Sentença",
        "Tempo Total", "Tempo por Sentença (s)", "Palavras por Segundo",
        "Uso de Memória (RAM)", "Memória GPU Alocada (MB)", "Memória GPU Reservada (MB)",
        "BLEU", "chr-F", "COMET", "BERTScore F1", "Erro"
    ]


def model_name_to_slug(model_name):
    """Nome do modelo para nome de arquivo (sem barras)."""
    return (model_name or "").replace("/", "_").replace("\\", "_").strip() or "unknown"

# -*- coding: utf-8 -*-
"""Preparação e carregamento dos datasets Scielo."""
import os
import csv
import random
from . import config


def load_scielo_csv(filepath):
    """Carrega CSV com abstracts."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
    
    abstracts = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        abstracts = list(reader)
    
    print(f"✅ Carregado: {len(abstracts):,} exemplos de {filepath}")
    return abstracts


def prepare_evaluation_csv(abstracts_file=config.SCIELO_ABSTRACTS_FILE,
                            train_csv=config.SCIELO_TRAIN_CSV,
                            val_csv=config.SCIELO_VAL_CSV,
                            test_csv=config.SCIELO_TEST_CSV,
                            train_samples=config.DEFAULT_TRAIN_SAMPLES,
                            val_samples=config.DEFAULT_VAL_SAMPLES,
                            test_samples=config.DEFAULT_TEST_SAMPLES):
    """
    Divide Scielo em TREINO/VALIDAÇÃO/TESTE com seed=42.
    
    ⭐ IMPORTANTE: Divisão feita ANTES de qualquer treino para evitar data leakage!
    
    Args:
        abstracts_file: CSV completo com todos os abstracts
        train_csv: Saída para TREINO
        val_csv: Saída para VALIDAÇÃO (internal monitoring)
        test_csv: Saída para TESTE (never before seen)
        train_samples: Número de exemplos para TREINO
        val_samples: Número de exemplos para VALIDAÇÃO
        test_samples: Número de exemplos para TESTE
    
    Returns:
        tuple: (train_csv, val_csv, test_csv) paths
    """
    
    print(f"\n[1/1] Dividindo Scielo em TREINO/VALIDAÇÃO/TESTE...")
    
    # Carregar abstracts completos
    abstracts = load_scielo_csv(abstracts_file)
    total_needed = train_samples + val_samples + test_samples
    
    print(f"    📊 Total disponível: {len(abstracts):,} exemplos")
    print(f"    📋 Total necessário: {total_needed:,} exemplos")
    
    # Ajustar se necessário
    if len(abstracts) < total_needed:
        print(f"    ⚠️  AVISO: Dataset tem menos exemplos que solicitado!")
        factor = len(abstracts) / total_needed
        train_samples = int(train_samples * factor)
        val_samples = int(val_samples * factor)
        test_samples = int(test_samples * factor)
        print(f"    Ajustado para: train={train_samples:,}, val={val_samples:,}, test={test_samples:,}")
    
    # Dividir com seed=42 para reprodutibilidade
    random.seed(config.SEED)
    random.shuffle(abstracts)
    
    # Selecionar subsets
    total_selected = train_samples + val_samples + test_samples
    selected = abstracts[:total_selected]
    
    train_data = selected[:train_samples]
    val_data = selected[train_samples:train_samples + val_samples]
    test_data = selected[train_samples + val_samples:train_samples + val_samples + test_samples]
    
    print(f"    ├─ 📚 TREINO: {len(train_data):,} exemplos → {train_csv}")
    print(f"    ├─ ✓ VALIDAÇÃO: {len(val_data):,} exemplos → {val_csv}")
    print(f"    └─ 🧪 TESTE: {len(test_data):,} exemplos → {test_csv}")
    
    # Salvar CSVs
    for data, filepath in [(train_data, train_csv), (val_data, val_csv), (test_data, test_csv)]:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["abstract_en", "abstract_pt"])
            writer.writeheader()
            writer.writerows(data)
    
    print(f"\n    ✅ Divisão completa!\n")
    
    return train_csv, val_csv, test_csv


def get_test_samples(test_csv=config.SCIELO_TEST_CSV, max_samples=None):
    """
    Carrega amostras de teste para avaliação.
    
    Args:
        test_csv: Path ao CSV de teste
        max_samples: Número máximo de amostras (None = todas)
    
    Returns:
        list: Dicts com 'abstract_en' e 'abstract_pt'
    """
    samples = load_scielo_csv(test_csv)
    
    if max_samples and len(samples) > max_samples:
        random.seed(config.SEED)
        samples = random.sample(samples, max_samples)
    
    return samples

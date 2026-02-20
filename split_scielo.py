#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Divide SciELO em 2 splits: TREINO (200k) e TESTE/VALIDAÇÃO (20k)."""
import csv
import random
import os

SEED = 42
ABSTRACTS_FILE = "finetuning/abstracts-datasets/abstracts_scielo.csv"
TRAIN_CSV = "finetuning/abstracts-datasets/scielo_abstracts_train.csv"
TEST_CSV = "finetuning/abstracts-datasets/scielo_abstracts_test.csv"

TRAIN_SAMPLES = 200_000
TEST_SAMPLES = 20_000

def split_scielo():
    """Divide SciELO em 2 splits com seed=42."""
    
    print(f"\n📊 Carregando índices de {ABSTRACTS_FILE}...")
    
    # Contar linhas (primeira passagem rápida)
    total_available = 0
    with open(ABSTRACTS_FILE, "r", encoding="utf-8") as f:
        total_available = sum(1 for _ in f) - 1  # -1 para header
    
    total_needed = TRAIN_SAMPLES + TEST_SAMPLES
    print(f"   ✓ Total disponível: {total_available:,} exemplos")
    print(f"   📋 Total necessário: {total_needed:,} exemplos")
    
    if total_available < total_needed:
        raise ValueError(f"Dataset tem {total_available:,} mas precisa de {total_needed:,}")
    
    # Criar índices aleatórios com seed
    print(f"\n🔀 Gerando índices com seed={SEED}...")
    indices = list(range(total_available))
    random.seed(SEED)
    random.shuffle(indices)
    
    # Selecionar índices para treino e teste
    train_indices = set(indices[:TRAIN_SAMPLES])
    test_indices = set(indices[TRAIN_SAMPLES:TRAIN_SAMPLES + TEST_SAMPLES])
    
    print(f"\n✂️  Dividindo:")
    print(f"   ├─ 📚 TREINO: {len(train_indices):,} exemplos")
    print(f"   └─ 🧪 TESTE: {len(test_indices):,} exemplos")
    
    # Salvar em uma única passagem
    os.makedirs(os.path.dirname(TRAIN_CSV), exist_ok=True)
    
    train_file = open(TRAIN_CSV, "w", newline="", encoding="utf-8")
    test_file = open(TEST_CSV, "w", newline="", encoding="utf-8")
    
    train_writer = csv.DictWriter(train_file, fieldnames=["abstract_en", "abstract_pt"])
    test_writer = csv.DictWriter(test_file, fieldnames=["abstract_en", "abstract_pt"])
    
    train_writer.writeheader()
    test_writer.writeheader()
    
    print(f"\n📝 Escrevendo arquivos...")
    
    row_idx = 0
    with open(ABSTRACTS_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row_idx in train_indices:
                train_writer.writerow(row)
            elif row_idx in test_indices:
                test_writer.writerow(row)
            
            if (row_idx + 1) % 100_000 == 0:
                print(f"   ✓ Processado: {row_idx + 1:,} linhas")
            row_idx += 1
    
    train_file.close()
    test_file.close()
    
    print(f"   ✅ {TRAIN_CSV}")
    print(f"   ✅ {TEST_CSV}")
    print(f"\n✓ Divisão completa!\n")

if __name__ == "__main__":
    split_scielo()

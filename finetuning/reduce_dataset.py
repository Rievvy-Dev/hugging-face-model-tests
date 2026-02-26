# -*- coding: utf-8 -*-
"""Script para reduzir dataset de treino para 10k."""
import csv
import random
from pathlib import Path

# Configuração
TRAIN_SIZE = 10000
VAL_SIZE = 2000
SEED = 42

# Paths
root = Path(__file__).parent / "abstracts-datasets"
train_path = root / "scielo_abstracts_train.csv"
val_path = root / "scielo_abstracts_val.csv"

# Carregar dados atuais
train_data = list(csv.DictReader(open(train_path, "r", encoding="utf-8")))
val_data = list(csv.DictReader(open(val_path, "r", encoding="utf-8")))

print(f"Dataset atual:")
print(f"  Train: {len(train_data):,}")
print(f"  Val: {len(val_data):,}")
print(f"  Total: {len(train_data) + len(val_data):,}")

# Combinar e embaralhar
all_data = train_data + val_data
random.seed(SEED)
random.shuffle(all_data)

# Dividir novamente
val_new = all_data[:VAL_SIZE]
train_new = all_data[VAL_SIZE:VAL_SIZE + TRAIN_SIZE]

# Salvar
with open(train_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["abstract_en", "abstract_pt"])
    writer.writeheader()
    writer.writerows(train_new)

with open(val_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["abstract_en", "abstract_pt"])
    writer.writeheader()
    writer.writerows(val_new)

print(f"\nNovo dataset:")
print(f"  Train: {len(train_new):,}")
print(f"  Val: {len(val_new):,}")
print(f"  Total: {len(train_new) + len(val_new):,}")
print(f"\n✅ Dataset reduzido com sucesso!")

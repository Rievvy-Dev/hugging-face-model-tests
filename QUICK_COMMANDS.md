# 🚀 Guia Rápido: 5 Estágios Simplificados

## 📋 O Fluxo é Simples

```
1. Buscar SciELO + Separar em train/val/test
   └─ python prepare_scielo_dataset.py
   └─ python finetuning/select_and_test_models.py  (separa automaticamente)

2. Testar os 2 MELHORES modelos na base SciELO
   └─ python finetuning/select_and_test_models.py --skip_prepare

3. Fine-tuning dos 2 modelos
   └─ python finetuning/finetune_selected_models.py --skip_prepare

4. Comparar antes/depois
   └─ python finetuning/select_and_test_models.py --test_both --skip_prepare
   └─ python compare_results.py
```

## ⚙️ Configurações Padrão

- **Batch Size (Treino)**: 2
- **Batch Size (Teste)**: 2
- **Épocas**: 5
- **Learning Rate**: 2e-5
- **Dataset Treino**: 200k exemplos
- **Dataset Teste**: 20k exemplos

💡 **Dica**: Use `--model helsinki` ou `--model m2m100` para testar/treinar modelos individualmente

---

## ⚡ Comandos Rápidos

### 1️⃣ Preparar Dataset
```bash
python prepare_scielo_dataset.py
```
Gera: `abstracts_scielo.csv` (2.7M exemplos)

---

### 2️⃣ Separar em Train/Val/Test
```bash
# Automático ao rodar (cria as 3 divisões)
python finetuning/select_and_test_models.py

# Ou via linha de comando
python -c "
from finetuning import datasets, config
datasets.prepare_evaluation_csv(
    abstracts_file='abstracts_scielo.csv',
    train_csv=config.SCIELO_TRAIN_CSV,
    val_csv=config.SCIELO_VAL_CSV,
    test_csv=config.SCIELO_TEST_CSV,
    train_samples=200000, val_samples=20000, test_samples=20000
)
"
```

Gera:
- `finetuning/abstracts-datasets/scielo_abstracts_train.csv` (200k)
- `finetuning/abstracts-datasets/scielo_abstracts_val.csv` (20k)
- `finetuning/abstracts-datasets/scielo_abstracts_test.csv` (20k)

---

### 3️⃣ Testar Modelos Base em SciELO
```bash
# Teste completo (20k amostras) - todos os modelos
python finetuning/select_and_test_models.py --skip_prepare

# Testar apenas Helsinki
python finetuning/select_and_test_models.py --skip_prepare --model helsinki

# Testar apenas M2M100
python finetuning/select_and_test_models.py --skip_prepare --model m2m100

# Teste rápido (5k amostras)
python finetuning/select_and_test_models.py --skip_prepare --test_samples 5000
```

Gera: `scielo_before_finetuning.csv`

---

### 4️⃣ Fine-tuning

#### Ambos os modelos
```bash
python finetuning/finetune_selected_models.py --skip_prepare
```

#### Modelo específico
```bash
python finetuning/finetune_selected_models.py --model helsinki --skip_prepare
python finetuning/finetune_selected_models.py --model m2m100 --skip_prepare
```

#### Com hiperparâmetros customizados
```bash
python finetuning/finetune_selected_models.py \
  --model helsinki \
  --epochs 10 \
  --batch_size 4 \
  --lr 5e-5 \
  --skip_prepare
```

#### Retomar se interrompido
```bash
python finetuning/finetune_selected_models.py \
  --model helena \
  --resume_from ./models/finetuned-scielo/helena/checkpoint-3000 \
  --skip_prepare
```

Gera: Modelos em `models/finetuned-scielo/`

---

### 5️⃣ Avaliar Fine-tuned e Comparar
```bash
# Testar fine-tuned (todos os modelos)
python finetuning/select_and_test_models.py --test_finetuned --skip_prepare

# Testar fine-tuned (modelo específico)
python finetuning/select_and_test_models.py --test_finetuned --model helsinki --skip_prepare

# Comparar base vs fine-tuned (todos)
python finetuning/select_and_test_models.py --test_both --skip_prepare

# Comparar base vs fine-tuned (modelo específico)
python finetuning/select_and_test_models.py --test_both --model m2m100 --skip_prepare

# Gerar relatório de comparação
python compare_results.py
```

Gera:
- `scielo_after_finetuning.csv`
- `SCIENCE_EVALUATION_REPORT.txt`

---

## 🔄 Pipeline Completo em Um Comando

```bash
python finetune_and_evaluate.py --skip_prepare
```

Executa STAGES 1-5 automaticamente (leva ~15-20 horas com GPU).

---

## 📊 Arquivos-Chave

| Arquivo | Função |
|---------|--------|
| `prepare_scielo_dataset.py` | Buscar SciELO |
| `finetuning/select_and_test_models.py` | Separar dados + Testar |
| `finetuning/finetune_selected_models.py` | Fine-tuning |
| `compare_results.py` | Gerar relatório |

---

Para mais detalhes, veja [README.md](README.md)

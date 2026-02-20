_# 📁 Estrutura Final do Projeto

## Organização Corrigida

A estrutura foi organizada para respeitar a **metodologia de 5 estágios**:

```
.
├── 📄 README.md                       ← LEIA ISTO PRIMEIRO!
├── 📄 QUICK_COMMANDS.md               ← Comandos rápidos
├── 📄 requirements.txt
├── 📄 requirements-ml.txt              
│
├── 🗂️ STAGE 0: Dataset
│   └── 🐍 prepare_scielo_dataset.py   (gera abstracts_scielo.csv)
│
├── 🗂️ STAGE 1: Avaliação Inicial (6 modelos × 4 datasets)
│   ├── 🐍 models-test.py              (5 modelos primários)
│   ├── 🐍 evaluate_quickmt.py         (6º modelo)
│   └── 📊 evaluation_results/
│       └── translation_metrics_all.csv ← RESULTADO
│
├── 🗂️ STAGE 2: Seleção dos Melhores
│   └── 🐍 choose_best_model.py        (Top 2 ranking)
│
├── 🗂️ STAGE 3: Preparação de Dados
│   └── 📦 finetuning/abstracts-datasets/
│       ├── scielo_abstracts_train.csv  (200k)
│       ├── scielo_abstracts_val.csv    (20k)
│       └── scielo_abstracts_test.csv   (20k)
│
├── 🗂️ STAGE 4: Fine-tuning
│   └── 🐍 finetuning/finetune_selected_models.py
│       └── 📂 models/finetuned-scielo/
│           ├── helena/
│           └── m2m100/
│
├── 🗂️ STAGE 5: Avaliação Final & Comparação
│   ├── 🐍 finetuning/select_and_test_models.py
│   ├── 🐍 compare_results.py
│   ├── 📊 evaluation_results/
│   │   ├── scielo_before_finetuning.csv
│   │   └── scielo_after_finetuning.csv
│   └── 📄 SCIENCE_EVALUATION_REPORT.txt ← RESULTADO FINAL
│
├── 🗂️ Módulos Core
│   └── 📦 finetuning/
│       ├── config.py              (configurações)
│       ├── models.py              (carregar/salvar)
│       ├── datasets.py            (preparação dados)
│       ├── metrics.py             (BLEU, chr-F, COMET, BERTScore)
│       ├── evaluate.py            (avaliação com progresso)
│       ├── trainer.py             (Seq2SeqTrainer)
│       ├── compare.py             (comparação)
│       ├── io_utils.py            (utilitários)
│       └── __init__.py
│
├── 🗂️ Checkpoints (para resumir se interrompido)
│   └── checkpoints/
│       ├── training/
│       └── evaluation/
│
├── 🗂️ Resultados de Avaliação Anterior
│   └── evaluation_results/
│       ├── translation_metrics_all.csv
│       ├── <modelo>.csv
│       └── [scielo_before/after_finetuning.csv]
│
└── 📊 Dataset Completo
    └── abstracts_scielo.csv   (2.7M exemplos)
```

---

## ✅ O que foi Restaurado/Mantido

| Arquivo | Status | Motivo |
|---------|--------|--------|
| `models-test.py` | ✅ Mantido | STAGE 1 - avalia 5 modelos |
| `evaluate_quickmt.py` | ✅ Mantido | STAGE 1 - avalia 6º modelo |
| `choose_best_model.py` | ✅ Mantido | STAGE 2 - seleciona top 2 |
| `prepare_scielo_dataset.py` | ✅ Mantido | STAGE 0 - gera dataset |
| `finetune_and_evaluate.py` | ✅ Mantido | Pipeline integrado (opcional) |
| `compare_results.py` | ✅ Mantido | STAGE 5 - relatório |
| `finetuning/select_and_test_models.py` | ✅ Novo | STAGE 3, 5 - testa em SciELO |
| `finetuning/finetune_selected_models.py` | ✅ Novo | STAGE 4 - fine-tuning |

---

## 📝 Metodologia Simplificada

```
1️⃣  Buscar + Separar dados SciELO
    ├─ prepare_scielo_dataset.py
    └─ select_and_test_models.py (cria train/val/test)

2️⃣  Testar modelos base em SciELO
    └─ select_and_test_models.py (gera scielo_before_finetuning.csv)

3️⃣  Fine-tune dos 2 modelos
    └─ finetune_selected_models.py (salva checkpoints)

4️⃣  Avaliar fine-tuned em SciELO
    └─ select_and_test_models.py --test_finetuned

5️⃣  Comparar base vs fine-tuned
    └─ compare_results.py (gera relatório)
```

---

## 🎯 Cronograma de Execução

```bash
# 1. Preparar dataset
python prepare_scielo_dataset.py                           # ~1 min

# 2. Separar dados (automático na próxima etapa)
# (vai ser criado por select_and_test_models.py)

# 3. Testar modelos base em SciELO
python finetuning/select_and_test_models.py --skip_prepare # ~3 horas

# 4. Fine-tuning (2 modelos × 5 épocas)
python finetuning/finetune_selected_models.py --skip_prepare  # ~8-12 horas

# 5. Avaliar e gerar relatório
python finetuning/select_and_test_models.py --test_both --skip_prepare  # ~3 horas
python compare_results.py                                 # ~10 seg

# Total: ~15-20 horas (com GPU)
```

---

## 🔑 Pontos-Chave da Metodologia

### ✅ Dados não se sobrepõem
- Train: 200k (74% dos 240k)
- Val: 20k (8%)
- Test: 20k (8% - MESMOS usados em STAGE 1!)

### ✅ Checkpoints permitem retomar
- STAGE 4: Checkpoints salvos a cada 1/5 da época
- STAGE 5: CSV armazenam estados intermediários

### ✅ Comparação justa
- STAGE 1: Testar modelos base nos 20k dados
- STAGE 5: Testar modelos fine-tuned nos MESMOS 20k dados
- Delta de BLEU mostra real melhoria

### ✅ Métricas compromete
- BLEU + chr-F (rápido, já calculado)
- COMET + BERTScore F1 (neural, mais preciso, mas lento)
- Score composto (0.30×BLEU + 0.25×chr-F + 0.25×COMET + 0.20×BS)

---

## 📚 Para Entender Melhor

1. **Leia primeiro**: [README.md](README.md) - explicação detalhada de cada estágio
2. **Comandos rápidos**: [QUICK_COMMANDS.md](QUICK_COMMANDS.md) - copy-paste dos comandos
3. **Ver configurações**: `finetuning/config.py` - ajustar hiperparâmetros
4. **Help dos scripts**:
   ```bash
   python finetuning/finetune_selected_models.py --help
   python finetuning/select_and_test_models.py --help
   ```

---

**Versão**: 3.0 | **Data**: Fevereiro 2026 | **Status**: ✅ Pronto para usar

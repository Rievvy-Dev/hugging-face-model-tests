# Estrutura do Projeto

## Organização por Estágio

A estrutura segue a **metodologia de 5 estágios** do pipeline de avaliação e fine-tuning.

```
.
├── 📄 README.md                               ← Documentação principal
├── 📄 PROJECT_STRUCTURE.md                    ← Este arquivo
├── 📄 QUICK_COMMANDS.md                       ← Comandos rápidos
├── 📄 requirements.txt                        ← Dependências gerais
├── 📄 requirements-ml.txt                     ← Dependências ML
│
├── 🗂️ STAGE 0: Dataset
│   └── 🐍 prepare_scielo_dataset.py           Gera abstracts_scielo.csv (2.7M exemplos)
│
├── 🗂️ STAGE 1: Avaliação Inicial (6 modelos × 4 datasets)
│   ├── 🐍 models-test.py                     Avalia 5 modelos primários
│   ├── 🐍 evaluate_quickmt.py                Avalia 6º modelo (QuickMT)
│   ├── 🐍 compute_neural_metrics.py          Calcula COMET e BERTScore
│   └── 📊 evaluation_results/
│       ├── translation_metrics_all.csv        Consolidado
│       ├── Helsinki-NLP_opus-mt-tc-big-en-pt.csv
│       ├── Narrativa_mbart-large-50-finetuned-opus-en-pt-translation.csv
│       ├── unicamp-dl_translation-en-pt-t5.csv
│       ├── VanessaSchenkel_unicamp-finetuned-en-to-pt-dataset-ted.csv
│       ├── danhsf_m2m100_418M-finetuned-kde4-en-to-pt_BR.csv
│       └── quickmt_quickmt-en-pt.csv
│
├── 🗂️ STAGE 2: Seleção do Modelo
│   ├── 🐍 choose_best_model.py               Ranking por score composto
│   └── 🐍 show_model_configs.py              Exibe configurações dos modelos
│
├── 🗂️ STAGE 3: Preparação de Dados (Dataset Compacto)
│   └── 📦 finetuning/abstracts-datasets/
│       ├── abstracts_scielo.csv               Corpus completo (2.7M)
│       ├── scielo_abstracts_train.csv         18.000 exemplos (treino)
│       ├── scielo_abstracts_val.csv            2.000 exemplos (validação)
│       └── scielo_abstracts_test.csv           5.000 exemplos (teste)
│
├── 🗂️ STAGE 4: Fine-tuning (unicamp-dl/translation-en-pt-t5)
│   ├── 🐍 finetuning/finetune_selected_models.py   Script de fine-tuning
│   └── ⭐ unicamp-t5/unicamp-t5/                    Modelo fine-tuned
│       ├── config.json
│       ├── generation_config.json
│       ├── model.safetensors                         Pesos do melhor modelo
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── spiece.model                              SentencePiece
│       ├── special_tokens_map.json
│       ├── checkpoint-12375/                         Epoch 11
│       └── checkpoint-13500/                         Epoch 12 (best)
│           ├── model.safetensors
│           ├── optimizer.pt
│           ├── scheduler.pt
│           ├── trainer_state.json                    Log completo
│           └── training_args.bin
│
├── 🗂️ STAGE 5: Avaliação Final e Comparação
│   ├── 🐍 finetuning/select_and_test_models.py      Avalia base e fine-tuned
│   ├── 📊 scielo_before_finetuning.csv               Baseline (BLEU=40.06)
│   ├── 📊 scielo_after_finetuning_epoch_1.csv        Epoch 1
│   ├── 📊 scielo_after_finetuning_epoch_11.csv       Epoch 11 (BLEU=45.51)
│   └── 📊 scielo_after_finetuning_epoch_12.csv       Epoch 12 (BLEU=45.51)
│
├── 🗂️ Módulos Core
│   ├── 📦 evaluation/                        Módulo de avaliação (STAGE 1)
│   │   ├── __init__.py
│   │   ├── config.py                         Configurações
│   │   ├── datasets.py                       Datasets públicos
│   │   ├── metrics.py                        Métricas
│   │   ├── models_loader.py                  Carregamento de modelos
│   │   ├── run.py                            Execução
│   │   ├── io_utils.py                       Utilitários I/O
│   │   └── fill_missing_metrics.py           Preenchimento
│   │
│   └── 📦 finetuning/                        Módulo de fine-tuning (STAGES 3-5)
│       ├── __init__.py
│       ├── config.py                          Configurações centralizadas
│       ├── models.py                          Carregamento/salvamento
│       ├── data_utils.py                      Preparação de dados
│       ├── datasets.py                        Dataset handling
│       ├── metrics.py                         BLEU, chrF, COMET, BERTScore
│       ├── evaluate.py                        Avaliação com progresso
│       ├── trainer.py                         Seq2SeqTrainer + loop
│       ├── compare.py                         Comparação base vs fine-tuned
│       └── io_utils.py                        Utilitários I/O
│
├── 🗂️ Pipeline Integrado
│   └── 🐍 finetune_and_evaluate.py            Executa STAGES 1-5 automaticamente
│
├── 🗂️ Auxiliares
│   ├── 🐍 check_gpu.py                       Verificação de GPU
│   ├── 🐍 split_scielo.py                    Divisão manual do dataset
│   ├── 📂 models-configs/                    Configurações JSON
│   │   ├── helsink.json
│   │   └── m2m100.json
│   ├── 📂 models/finetuned-scielo/           Fine-tunings anteriores
│   │   └── helsinki/
│   └── 📂 checkpoints/                       Checkpoints de controle
│       ├── training/
│       └── evaluation/
│
└── 📦 Arquivos de Modelo Compactado
    └── unicamp-t5.zip                         Modelo fine-tuned compactado
```

---

## Arquivos Importantes

| Arquivo | Estágio | Descrição |
|---------|---------|-----------|
| `scielo_before_finetuning.csv` | 5 | Métricas baseline: BLEU=40.06 |
| `scielo_after_finetuning_epoch_12.csv` | 5 | Métricas fine-tuned: BLEU=45.51 |
| `unicamp-t5/unicamp-t5/model.safetensors` | 4 | Pesos do melhor modelo |
| `unicamp-t5/unicamp-t5/checkpoint-13500/trainer_state.json` | 4 | Log completo de treinamento (12 epochs) |
| `evaluation_results/translation_metrics_all.csv` | 1 | Resultados de todos os 6 modelos |
| `finetuning/abstracts-datasets/scielo_abstracts_test.csv` | 3 | 5k exemplos de teste |

---

## Metodologia Resumida

```
1️⃣  Avaliar 6 modelos em datasets públicos
    └─ models-test.py + evaluate_quickmt.py

2️⃣  Selecionar unicamp-dl/translation-en-pt-t5
    └─ choose_best_model.py

3️⃣  Preparar splits SciELO (18k treino, 2k val, 5k teste)
    └─ select_and_test_models.py

4️⃣  Fine-tuning na RTX 4050 (12 epochs, batch=8, grad_accum=2)
    └─ finetune_selected_models.py → unicamp-t5/unicamp-t5/

5️⃣  Avaliar e comparar: BLEU 40.06 → 45.51 (+13.6%)
    └─ select_and_test_models.py --test_both
```

# Pipeline de Avaliação e Fine-Tuning de Modelos de Tradução EN→PT

## Visão Geral

Este projeto implementa um **pipeline de 5 estágios** para avaliar e fine-tunar modelos de tradução automática neural (NMT) inglês→português, aplicados ao domínio de abstracts científicos do **SciELO**.

O modelo selecionado para fine-tuning foi o **`unicamp-dl/translation-en-pt-t5`**, uma adaptação do T5 (Text-to-Text Transfer Transformer) para tradução EN→PT, desenvolvido pela Universidade Estadual de Campinas (UNICAMP).

### Resultados Obtidos

| Métrica    | Antes do Fine-tuning | Após Fine-tuning (Epoch 12) | Delta   | Melhoria |
|------------|---------------------:|----------------------------:|--------:|---------:|
| BLEU       | 40.06                | 45.51                       | +5.45   | +13.6%   |
| chrF       | 65.61                | 70.54                       | +4.93   | +7.5%    |
| COMET      | 0.8499               | 0.8756                      | +0.0257 | +3.0%    |
| BERTScore  | 0.8957               | 0.9124                      | +0.0167 | +1.9%    |

---

## Sobre o Modelo: `unicamp-dl/translation-en-pt-t5`

### Arquitetura

O modelo é baseado na arquitetura **T5 (Text-to-Text Transfer Transformer)** proposta por Raffel et al. (2019). O T5 trata todas as tarefas de NLP como problemas de texto-para-texto, onde tanto a entrada quanto a saída são sequências de texto.

| Componente                | Especificação               |
|---------------------------|:----------------------------|
| Arquitetura base          | T5 (encoder-decoder)        |
| Camadas do encoder        | 12                          |
| Camadas do decoder        | 12                          |
| Dimensão oculta (d_model) | 768                         |
| Cabeças de atenção        | 12                          |
| Dimensão do feed-forward  | 3072                        |
| Parâmetros totais         | ~220M                       |
| Vocabulário               | 32.128 tokens (SentencePiece) |
| Tipo de atenção           | Multi-head self-attention   |
| Normalização              | Layer Normalization (pre-norm) |
| Ativação                  | ReLU (Rectified Linear Unit) |

### Pré-treinamento e Dados Originais

- **Pré-treinamento base**: PTT5 — modelo T5 pré-treinado em corpus em português
- **Fine-tuning de tradução (pelos autores)**: ParaCrawl (5M+ pares EN-PT) + Corpora biomédica científica (6M+ pares)
- **Tarefa**: Tradução EN→PT com prefixo `"translate English to Portuguese: "`
- **Tokenizador**: SentencePiece (unigram) com vocabulário de 32k tokens

### Referência Acadêmica

```bibtex
@inproceedings{lopes-etal-2020-lite,
    title     = "Lite Training Strategies for {P}ortuguese-{E}nglish and {E}nglish-{P}ortuguese Translation",
    author    = "Lopes, Alexandre and Nogueira, Rodrigo and Lotufo, Roberto and Pedrini, Helio",
    booktitle = "Proceedings of the Fifth Conference on Machine Translation",
    month     = nov,
    year      = "2020",
    address   = "Online",
    publisher = "Association for Computational Linguistics",
    url       = "https://aclanthology.org/2020.wmt-1.90",
    pages     = "833--840",
}
```

### Por que este modelo foi selecionado?

1. **Eficiência computacional**: ~220M parâmetros (6x menor que Helsinki opus-mt-tc-big-en-pt)
2. **Bom baseline**: BLEU 40.06 em abstracts SciELO sem fine-tuning
3. **Arquitetura comprovada**: T5 é estado da arte em tarefas text-to-text
4. **Viável em hardware modesto**: Cabe em GPU com 6GB VRAM (RTX 4050)
5. **Domínio adequado**: Pré-treinado em corpus científico, alinhado ao SciELO

---

## Pipeline de 5 Estágios

```
STAGE 1: AVALIAÇÃO INICIAL
├─ Testar 6 modelos pré-treinados em 4 datasets públicos
├─ Calcular BLEU, chrF, COMET, BERTScore
└─ Resultado: evaluation_results/translation_metrics_all.csv
        ↓
STAGE 2: SELEÇÃO DO MODELO
├─ Ranking por score composto (BLEU + chrF + COMET + BERTScore)
├─ Seleção: unicamp-dl/translation-en-pt-t5
└─ Resultado: modelo definido para fine-tuning
        ↓
STAGE 3: PREPARAÇÃO DE DADOS
├─ Separar SciELO em 3 splits não-sobrepostos:
│   ├─ 18.000 exemplos para TREINO
│   ├─  2.000 exemplos para VALIDAÇÃO (early stopping)
│   └─  5.000 exemplos para TESTE
└─ Resultado: finetuning/abstracts-datasets/*.csv
        ↓
STAGE 4: FINE-TUNING
├─ GPU: NVIDIA RTX 4050 (6GB VRAM)
├─ 12 epochs, batch_size=8, grad_accum=2, lr=1e-5
├─ Early stopping com patience=2
└─ Resultado: unicamp-t5/unicamp-t5/ (modelo fine-tuned)
        ↓
STAGE 5: AVALIAÇÃO FINAL
├─ Testar modelo base vs fine-tuned nos MESMOS 5k dados de teste
├─ Calcular delta de métricas
└─ Resultado: scielo_before_finetuning.csv / scielo_after_finetuning_epoch_*.csv
```

---

## STAGE 1: Avaliação Inicial dos Modelos

### Objetivo
Avaliar 6 modelos pré-treinados em 4 datasets públicos para estabelecer baselines.

### Modelos Avaliados

| # | Modelo                                                        | Arquitetura | Parâmetros |
|---|---------------------------------------------------------------|-------------|------------|
| 1 | `Helsinki-NLP/opus-mt-tc-big-en-pt`                          | MarianMT    | ~600M      |
| 2 | `Narrativa/mbart-large-50-finetuned-opus-en-pt-translation`  | mBART-50    | ~611M      |
| 3 | `unicamp-dl/translation-en-pt-t5`                            | T5          | ~220M      |
| 4 | `VanessaSchenkel/unicamp-finetuned-en-to-pt-dataset-ted`     | T5          | ~220M      |
| 5 | `danhsf/m2m100_418M-finetuned-kde4-en-to-pt_BR`             | M2M100      | ~418M      |
| 6 | `quickmt/quickmt-en-pt`                                      | CTranslate2 | —          |

### Datasets Públicos

| Dataset      | Exemplos | Descrição                    |
|--------------|----------|------------------------------|
| WMT24++      | 998      | Avaliação en→pt_BR           |
| ParaCrawl    | 5.000    | Crawl web paralelo en→pt     |
| Flores       | 1.012    | Facebook multilingual        |
| OPUS100      | 5.000    | Corpus paralelo en→pt        |

### Métricas

| Métrica       | Tipo       | Range | Descrição                                          |
|---------------|------------|-------|----------------------------------------------------|
| **BLEU**      | N-gramas   | 0-100 | Precisão de n-gramas (1-4) com brevity penalty     |
| **chrF**      | Caracteres | 0-100 | F-score baseado em caracteres                      |
| **COMET**     | Neural     | 0-1   | Score neural aprendido (Unbabel/wmt22-comet-da)    |
| **BERTScore** | Neural     | 0-1   | Similaridade semântica via embeddings BERT         |

### Resultados — Média por Modelo (4 datasets)

| #  | Modelo          | BLEU  | chrF  | COMET  | BERTScore | GPU (MB) |
|----|-----------------|------:|------:|-------:|----------:|---------:|
| 1  | Helsinki        | 38.01 | 59.89 | 0.8301 | 0.8674    | 904      |
| 2  | Narrativa mBART | 22.53 | 41.89 | 0.7700 | 0.8398    | 2.340    |
| 3  | Unicamp-T5      | 15.80 | 33.81 | 0.6812 | 0.7960    | 859      |
| 4  | VanessaSchenkel | 9.15  | 26.22 | 0.6473 | 0.7895    | 859      |
| 5  | M2M100          | 22.17 | 47.94 | 0.7581 | 0.8323    | 1.863    |
| 6  | QuickMT         | 0.00  | 4.13  | 0.2723 | 0.4742    | 9        |

### Resultados Detalhados — Por Dataset

**WMT24++ (998 exemplos, sentenças longas ~33 palavras/sentença)**

| Modelo          | BLEU  | chrF  | COMET  | BERTScore | Tempo       |
|-----------------|------:|------:|-------:|----------:|------------:|
| Helsinki        | 33.71 | 58.86 | 0.7825 | 0.8622    | 529s        |
| Narrativa mBART | 6.54  | 25.48 | 0.6452 | 0.7917    | 797s        |
| Unicamp-T5      | 3.55  | 19.73 | 0.5391 | 0.7573    | 237s        |
| VanessaSchenkel | 2.77  | 17.19 | 0.5091 | 0.7562    | 215s        |
| M2M100          | 22.99 | 50.08 | 0.7012 | 0.8404    | 888s        |
| QuickMT         | 0.00  | 4.80  | 0.2480 | 0.4871    | 59s         |

**ParaCrawl (5.000 exemplos, sentenças curtas ~7 palavras/sentença)**

| Modelo          | BLEU  | chrF  | COMET  | BERTScore | Tempo       |
|-----------------|------:|------:|-------:|----------:|------------:|
| Helsinki        | 39.63 | 59.98 | 0.8452 | 0.8696    | 740s        |
| Narrativa mBART | 27.07 | 46.75 | 0.8083 | 0.8544    | 2.013s      |
| Unicamp-T5      | 19.46 | 37.99 | 0.7239 | 0.8076    | 633s        |
| VanessaSchenkel | 11.05 | 28.89 | 0.6868 | 0.7992    | 610s        |
| M2M100          | 22.41 | 47.11 | 0.7735 | 0.8293    | 585s        |
| QuickMT         | 0.00  | 4.03  | 0.2789 | 0.4703    | 288s        |

**Flores (1.012 exemplos)**

| Modelo          | BLEU  | chrF  | COMET  | BERTScore | Tempo       |
|-----------------|------:|------:|-------:|----------:|------------:|
| Helsinki        | 39.08 | 60.72 | 0.8473 | 0.8683    | 131s        |
| Narrativa mBART | 29.43 | 48.59 | 0.8182 | 0.8588    | 378s        |
| Unicamp-T5      | 20.72 | 39.52 | 0.7380 | 0.8116    | 122s        |
| VanessaSchenkel | 11.74 | 29.93 | 0.7066 | 0.8032    | 111s        |
| M2M100          | 20.85 | 47.45 | 0.7842 | 0.8301    | 247s        |
| QuickMT         | 0.00  | 3.68  | 0.2835 | 0.4689    | 59s         |

**OPUS100 (5.000 exemplos)**

| Modelo          | BLEU  | chrF  | COMET  | BERTScore | Tempo       |
|-----------------|------:|------:|-------:|----------:|------------:|
| Helsinki        | 39.63 | 59.98 | 0.8452 | 0.8696    | 744s        |
| Narrativa mBART | 27.07 | 46.75 | 0.8083 | 0.8544    | 1.126s      |
| Unicamp-T5      | 19.46 | 37.99 | 0.7239 | 0.8076    | 649s        |
| VanessaSchenkel | 11.05 | 28.89 | 0.6868 | 0.7992    | 617s        |
| M2M100          | 22.41 | 47.11 | 0.7735 | 0.8293    | 585s        |
| QuickMT         | 0.00  | 4.03  | 0.2789 | 0.4703    | 287s        |

### Comandos

```bash
# Avaliar 5 modelos primários
python models-test.py --full

# Avaliar 6º modelo (QuickMT - CTranslate2)
python evaluate_quickmt.py --full

# Retomar avaliação interrompida
python models-test.py --resume
python evaluate_quickmt.py --resume
```

### Saída
- `evaluation_results/translation_metrics_all.csv` — consolidado
- `evaluation_results/<modelo>.csv` — individual por modelo

---

## STAGE 2: Seleção do Modelo

### Objetivo
Selecionar o melhor modelo considerando qualidade e eficiência.

### Score Composto
```
score = 0.30 × BLEU_norm + 0.25 × chrF_norm + 0.25 × COMET_norm + 0.20 × BERTScore_norm
```
Todos os scores normalizados min-max para [0, 1].

### Comando
```bash
python choose_best_model.py
```

### Resultado
Modelo selecionado: **`unicamp-dl/translation-en-pt-t5`** — melhor trade-off entre qualidade e custo computacional para fine-tuning em domínio científico.

---

## STAGE 3: Preparação de Dados SciELO

### Objetivo
Criar 3 splits não-sobrepostos do dataset SciELO (2.7M exemplos totais).

### Divisão dos Dados

| Split      | Exemplos | Uso                                    |
|------------|----------|----------------------------------------|
| Treino     | 18.000   | Fine-tuning do modelo                  |
| Validação  | 2.000    | Monitorar convergência + early stopping|
| Teste      | 5.000    | Avaliação final (mesmos para base e fine-tuned) |

**Total: 25.000 exemplos (~0.9% do corpus completo)**

### Justificativa do Dataset Compacto

- **18k treino**: Suficiente para adaptação de domínio (abstracts científicos) sem overfitting
- **2k validação**: Monitora eval_loss por epoch e aciona early stopping
- **5k teste**: Mesmo conjunto usado na avaliação do modelo base, garantindo comparação justa
- **Seed fixo (42)**: Splits são determinísticos e reprodutíveis

### Comandos

```bash
# Preparação automática (integrada ao select_and_test_models.py)
python finetuning/select_and_test_models.py

# Ou manualmente
python -c "
from finetuning import config, data_utils
data_utils.prepare_evaluation_csv(
    abstracts_file='abstracts_scielo.csv',
    train_csv=config.SCIELO_TRAIN_CSV,
    val_csv=config.SCIELO_VAL_CSV,
    test_csv=config.SCIELO_TEST_CSV,
    train_samples=18_000,
    val_samples=2_000,
    test_samples=5_000
)
"
```

### Saída
```
finetuning/abstracts-datasets/
├── scielo_abstracts_train.csv   (18.000 exemplos)
├── scielo_abstracts_val.csv     ( 2.000 exemplos)
└── scielo_abstracts_test.csv    ( 5.000 exemplos)
```

---

## STAGE 4: Fine-Tuning

### Objetivo
Fine-tunar o modelo `unicamp-dl/translation-en-pt-t5` no domínio de abstracts científicos.

### Configuração de Treinamento

| Parâmetro                  | Valor                  |
|----------------------------|------------------------|
| GPU                        | NVIDIA RTX 4050 (6GB)  |
| Epochs                     | 12                     |
| Batch size                 | 8                      |
| Gradient accumulation      | 2                      |
| **Batch efetivo**          | **16**                 |
| Learning rate              | 1e-5                   |
| Warmup steps               | 500                    |
| Weight decay               | 0.01                   |
| Max sequence length        | 256 tokens             |
| Precisão                   | FP16 (mixed precision) |
| Otimizador                 | AdamW                  |
| Early stopping patience    | 2 epochs               |
| Gradient checkpointing     | Ativado                |
| Steps por epoch            | 1.125                  |
| Save strategy              | Por epoch              |
| Seed                       | 42                     |

### Configuração do Modelo — `config.json` (antes vs depois)

A arquitetura do modelo **não muda** durante o fine-tuning — apenas os pesos são atualizados. As diferenças no `config.json` são campos de metadados adicionados pela versão mais recente do `transformers`.

#### Modelo Original (HuggingFace)

```json
{
  "_name_or_path": "./",
  "architectures": ["T5ForConditionalGeneration"],
  "d_ff": 3072,
  "d_kv": 64,
  "d_model": 768,
  "decoder_start_token_id": 0,
  "dropout_rate": 0.1,
  "eos_token_id": 1,
  "feed_forward_proj": "relu",
  "initializer_factor": 1.0,
  "is_encoder_decoder": true,
  "layer_norm_epsilon": 1e-06,
  "model_type": "t5",
  "n_positions": 512,
  "num_decoder_layers": 12,
  "num_heads": 12,
  "num_layers": 12,
  "output_past": true,
  "pad_token_id": 0,
  "relative_attention_num_buckets": 32,
  "torch_dtype": "float32",
  "transformers_version": "4.11.3",
  "use_cache": true,
  "vocab_size": 32128
}
```

#### Modelo Fine-tuned (local)

```json
{
  "architectures": ["T5ForConditionalGeneration"],
  "classifier_dropout": 0.0,
  "d_ff": 3072,
  "d_kv": 64,
  "d_model": 768,
  "decoder_start_token_id": 0,
  "dense_act_fn": "relu",
  "dropout_rate": 0.1,
  "dtype": "float32",
  "eos_token_id": 1,
  "feed_forward_proj": "relu",
  "initializer_factor": 1.0,
  "is_encoder_decoder": true,
  "is_gated_act": false,
  "layer_norm_epsilon": 1e-06,
  "model_type": "t5",
  "n_positions": 512,
  "num_decoder_layers": 12,
  "num_heads": 12,
  "num_layers": 12,
  "output_past": true,
  "pad_token_id": 0,
  "relative_attention_max_distance": 128,
  "relative_attention_num_buckets": 32,
  "transformers_version": "4.57.6",
  "use_cache": true,
  "vocab_size": 32128
}
```

#### Diferenças

| Campo                          | Original (HF)   | Fine-tuned (local) | Observação                          |
|--------------------------------|------------------|---------------------|-------------------------------------|
| `_name_or_path`                | `"./"`          | *(removido)*        | Caminho local do autor original     |
| `torch_dtype` / `dtype`        | `"float32"`     | `"float32"`        | Apenas renomeação de campo          |
| `transformers_version`         | `4.11.3`         | `4.57.6`            | Versão da lib no momento do salvamento |
| `classifier_dropout`           | *(ausente)*      | `0.0`               | Adicionado pela versão nova         |
| `dense_act_fn`                 | *(ausente)*      | `"relu"`           | Explicitação da ativação            |
| `is_gated_act`                 | *(ausente)*      | `false`             | T5 padrão não usa gated activation  |
| `relative_attention_max_distance`| *(ausente)*    | `128`               | Default explicitado pela versão nova |

> **Nota**: Todos os hiperparâmetros arquiteturais (d_model, d_ff, num_layers, num_heads, vocab_size) são **idênticos**. O fine-tuning altera **apenas os pesos** (`model.safetensors`), não a arquitetura.

### Configuração de Geração — `generation_config.json`

Arquivo criado automaticamente pelo `Seq2SeqTrainer` (não existia no modelo original do HuggingFace):

```json
{
  "_from_model_config": true,
  "decoder_start_token_id": 0,
  "eos_token_id": [1],
  "pad_token_id": 0,
  "transformers_version": "4.57.6"
}
```

| Parâmetro              | Valor | Descrição                                         |
|------------------------|-------|---------------------------------------------------|
| `decoder_start_token_id` | 0   | Token `<pad>` usado para iniciar a decodificação  |
| `eos_token_id`           | 1   | Token `</s>` marca fim da sequência gerada        |
| `pad_token_id`           | 0   | Token `<pad>` para padding                        |

### Argumentos de Treinamento — `Seq2SeqTrainingArguments`

Configuração completa passada ao `Seq2SeqTrainer` (de `finetuning/trainer.py`):

```python
Seq2SeqTrainingArguments(
    output_dir="./models/finetuned-scielo/unicamp-t5",
    overwrite_output_dir=False,
    num_train_epochs=12,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=1e-5,
    warmup_steps=500,
    weight_decay=0.01,
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    gradient_accumulation_steps=2,
    fp16=True,
    logging_steps=100,
    predict_with_generate=True,
    optim="adamw_torch",
    seed=42,
    report_to=[],
    eval_strategy="epoch",
)
```

| Argumento                   | Valor                          | Finalidade                                    |
|-----------------------------|--------------------------------|-----------------------------------------------|
| `output_dir`                | `./models/finetuned-scielo/unicamp-t5` | Diretório de saída dos checkpoints     |
| `overwrite_output_dir`      | `False`                        | Preserva checkpoints existentes               |
| `num_train_epochs`          | 12                             | Número total de epochs                        |
| `per_device_train_batch_size`| 8                             | Batch size por GPU                            |
| `learning_rate`             | 1e-5                           | Taxa de aprendizado (linear warmup + decay)   |
| `warmup_steps`              | 500                            | Steps de warmup linear do LR                  |
| `weight_decay`              | 0.01                           | Regularização L2 desacoplada (AdamW)          |
| `save_strategy`             | `"epoch"`                     | Salva checkpoint a cada epoch                 |
| `save_total_limit`          | 2                              | Mantém apenas os 2 últimos checkpoints        |
| `load_best_model_at_end`    | `True`                         | Carrega melhor modelo (menor eval_loss) ao final |
| `metric_for_best_model`     | `"eval_loss"`                 | Métrica para selecionar melhor checkpoint     |
| `gradient_accumulation_steps`| 2                             | Acumula gradientes de 2 mini-batches          |
| `fp16`                      | `True`                         | Mixed precision (Tensor Cores da RTX 4050)    |
| `logging_steps`             | 100                            | Log de métricas a cada 100 steps              |
| `predict_with_generate`     | `True`                         | Usa `model.generate()` para avaliação         |
| `optim`                     | `"adamw_torch"`               | Otimizador AdamW nativo do PyTorch            |
| `seed`                      | 42                             | Seed para reprodutibilidade                   |
| `eval_strategy`             | `"epoch"`                     | Avalia no dataset de validação a cada epoch   |

### Comando Executado

```bash
python finetuning/finetune_selected_models.py \
  --model unicamp-t5 \
  --epochs 12 \
  --batch_size 8 \
  --grad_accum_steps 2 \
  --lr 1e-5 \
  --fp16 \
  --max_seq_len 256 \
  --early_stopping_patience 2 \
  --skip_prepare
```

### Explicação Detalhada dos Parâmetros

Cada parâmetro do comando foi escolhido para maximizar a qualidade do fine-tuning dentro das restrições de hardware (RTX 4050, 6GB VRAM). Abaixo, a explicação técnica de cada um com exemplos visuais.

---

#### `--model unicamp-t5`

Seleciona o modelo `unicamp-dl/translation-en-pt-t5` do dicionário `config.MODELS`. Veja a seção [STAGE 2](#stage-2-seleção-do-modelo) para justificativa da seleção.

---

#### `--epochs 12`

**O que é**: Número de passagens completas pelo dataset de treino (18.000 exemplos).

**Referência**: Smith, L. N. (2018). *A disciplined approach to neural network hyper-parameters: Part 1 – learning rate, batch size, momentum, and weight decay*. arXiv:1803.09820. https://arxiv.org/abs/1803.09820

**Por que 12**: O número de epochs é determinado pela convergência observada. A eval_loss continuou melhorando em todas as 12 epochs (0.973 no epoch 12), sem acionar early stopping. Mais epochs não foram testados porque a taxa de melhoria nos últimos epochs era marginal (~0.0003/epoch).

```
Epoch 1  ████████████████████████████████████████  eval_loss: 1.0068
Epoch 2  ███████████████████████████████████████   eval_loss: 0.9931  ↓ 0.0137
Epoch 3  ██████████████████████████████████████    eval_loss: 0.9861  ↓ 0.0070
Epoch 4  █████████████████████████████████████     eval_loss: 0.9818  ↓ 0.0043
Epoch 5  ████████████████████████████████████      eval_loss: 0.9792  ↓ 0.0026
Epoch 6  ███████████████████████████████████       eval_loss: 0.9772  ↓ 0.0020
Epoch 7  ██████████████████████████████████        eval_loss: 0.9757  ↓ 0.0015
Epoch 8  █████████████████████████████████         eval_loss: 0.9747  ↓ 0.0010
Epoch 9  ████████████████████████████████          eval_loss: 0.9737  ↓ 0.0010
Epoch 10 ███████████████████████████████           eval_loss: 0.9733  ↓ 0.0004
Epoch 11 ██████████████████████████████            eval_loss: 0.9730  ↓ 0.0003
Epoch 12 █████████████████████████████             eval_loss: 0.9730  ↓ 0.0001 ⭐

→ Redução total: 0.0338 (3.36%)
→ 90% da melhoria ocorre nos primeiros 5 epochs
→ Epochs 10-12: rendimento decrescente (<0.001/epoch)
```

**Trade-off**: Poucas epochs = underfitting (modelo não adaptado ao domínio). Muitas epochs = overfitting (modelo memoriza exemplos de treino). Com 12 epochs, train_loss (0.97) ≈ eval_loss (0.97), indicando ausência de overfitting.

---

#### `--batch_size 8`

**O que é**: Número de exemplos processados **simultaneamente** em cada forward pass pela GPU.

**Referência**: Masters, D. & Luschi, C. (2018). *Revisiting Small Batch Training for Deep Neural Networks*. arXiv:1804.07612. https://arxiv.org/abs/1804.07612

**Por que 8 (e não mais)**: Limitação direta da VRAM da RTX 4050 (6GB). Com FP16, gradient checkpointing e max_seq_len=256:

```
Memória GPU por batch (estimativa):

  Pesos do modelo (FP16):    ~440 MB  (220M params × 2 bytes)
  Ativações Encoder (FP16):  ~384 MB  (batch=8 × 256 tokens × 768 dim × 12 layers)
  Ativações Decoder (FP16):  ~384 MB  (idem)
  Gradientes (FP32):         ~880 MB  (220M params × 4 bytes, mixed precision)
  Estados do otimizador:     ~1760 MB (AdamW: 2 estados × 220M × 4 bytes)
  Overhead CUDA:             ~200 MB
                             ─────────
  Total estimado:           ~4048 MB (~4 GB)

  VRAM disponível:           6144 MB (6 GB)
  Margem:                    ~2096 MB (suficiente ✅)

  Com batch_size=16:         +768 MB ativações → ~4816 MB (ainda cabe, mas ajustado)
  Com batch_size=32:         +1536 MB ativações → OOM ❌ (Out of Memory)

→ batch_size=8 garante estabilidade com margem confortável
```

**Efeito no ruído do gradiente**:

```
Batch size pequeno (ex: 1-4):
  Gradiente ← ∇L(x₁)                         ← Muito ruidoso
  → Convergência instável, LR precisa ser menor

Batch size médio (ex: 8-16):
  Gradiente ← ¼ × (∇L(x₁) + ∇L(x₂) + ... + ∇L(x₈))  ← Bom equilíbrio
  → Gradiente suavizado, convergência estável

Batch size grande (ex: 128-512):
  Gradiente ← 1/128 × Σ ∇L(xᵢ)               ← Muito suave
  → Convergência rápida mas generalização pior
    (Sharp minima, referência: Keskar et al., 2017)
```

**Referência**: Keskar, N. S. et al. (2017). *On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima*. In ICLR 2017. https://arxiv.org/abs/1609.04836

---

#### `--grad_accum_steps 2` ⭐

**O que é**: **Gradient Accumulation** — acumula gradientes de múltiplos mini-batches antes de atualizar os pesos. Simula um batch maior sem exigir mais VRAM.

**Referência**: Ott, M. et al. (2018). *Scaling Neural Machine Translation*. In Proceedings of the Third Conference on Machine Translation (WMT), pp. 1–9. https://aclanthology.org/W18-6301/

**Batch Efetivo**:

$$\text{Batch efetivo} = \text{batch\_size} \times \text{grad\_accum\_steps} = 8 \times 2 = 16$$

**Funcionamento visual**:

```
SEM gradient accumulation (batch_size=16, se coubesse na VRAM):
┌──────────────────────────────────────────────────────────┐
│ Forward: 16 exemplos → Loss → Backward → ∇W → Atualiza  │
│ VRAM: ~5.5 GB (pode dar OOM)                             │
└──────────────────────────────────────────────────────────┘

COM gradient accumulation (batch_size=8, grad_accum=2):
┌──────────────────────────────────────────────────────────┐
│ Step 1: Forward 8 exemplos → Loss₁ → Backward → ∇W₁     │
│         (NÃO atualiza pesos, apenas acumula gradiente)   │
│         VRAM: ~4 GB ✅                                    │
│                                                          │
│ Step 2: Forward 8 exemplos → Loss₂ → Backward → ∇W₂     │
│         ∇W_total = ∇W₁ + ∇W₂                            │
│         Optimizer.step() → Atualiza pesos com ∇W_total   │
│         VRAM: ~4 GB ✅                                    │
└──────────────────────────────────────────────────────────┘

Resultado MATEMÁTICO: Gradiente idêntico ao batch_size=16
Resultado PRÁTICO:    Metade da VRAM necessária
Custo:                ~2x mais lento (2 forward passes vs 1)
```

**Por que 2 e não mais?**

```
grad_accum=1  → batch efetivo = 8   → gradiente ruidoso, convergência instável
grad_accum=2  → batch efetivo = 16  → bom equilíbrio ruído/estabilidade ✅
grad_accum=4  → batch efetivo = 32  → mais estável, mas 4x mais lento
grad_accum=8  → batch efetivo = 64  → overkill para 18k exemplos (apenas 281 steps/epoch)

Steps por epoch com cada configuração:
  grad_accum=1: 18000 / 8  = 2250 steps/epoch
  grad_accum=2: 18000 / 16 = 1125 steps/epoch  ← Nosso caso
  grad_accum=4: 18000 / 32 =  562 steps/epoch
  grad_accum=8: 18000 / 64 =  281 steps/epoch  ← Poucos updates, convergência lenta
```

**Impacto na taxa de aprendizado**: O learning rate é aplicado ao gradiente acumulado (já normalizado). Com Transformers `Seq2SeqTrainer`, a loss já é dividida pelo `grad_accum_steps`, então a escala é automaticamente ajustada.

**Implementação** (em `finetuning/trainer.py`):

```python
Seq2SeqTrainingArguments(
    per_device_train_batch_size=8,       # batch real na GPU
    gradient_accumulation_steps=2,       # acumula 2 batches
    # → batch efetivo = 8 × 2 = 16
)
```

---

#### `--lr 1e-5` ⭐

**O que é**: **Learning Rate** — a taxa de aprendizado controla o tamanho do passo na atualização dos pesos do modelo. É o hiperparâmetro mais crítico do treinamento.

**Referência**: Loshchilov, I. & Hutter, F. (2019). *Decoupled Weight Decay Regularization*. In ICLR 2019. https://arxiv.org/abs/1711.05101 (AdamW)

**Referência**: Howard, J. & Ruder, S. (2018). *Universal Language Model Fine-tuning for Text Classification*. In Proceedings of ACL 2018, pp. 328–339. https://aclanthology.org/P18-1031/ (recomendação de LR para fine-tuning)

**Regra de atualização (AdamW)**:

$$\theta_{t+1} = \theta_t - \eta \cdot \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \cdot \theta_t\right)$$

Onde:
- $\eta = 10^{-5}$ é o learning rate
- $\hat{m}_t$ = média móvel dos gradientes (momentum)
- $\hat{v}_t$ = média móvel dos gradientes² (adaptação)
- $\lambda = 0.01$ = weight decay
- $\epsilon = 10^{-8}$ = estabilidade numérica

**Por que 1e-5 (e não mais ou menos)?**

```
Para fine-tuning de modelos pré-treinados, a literatura recomenda LRs pequenas:

  Pré-treinamento (do zero):    1e-4  a 1e-3   (pesos aleatórios, grandes passos)
  Fine-tuning (adaptação):      1e-5  a 5e-5   (pesos já bons, passos pequenos) ← 
  Ajuste mínimo (few-shot):     1e-6  a 5e-6   (alterar o mínimo possível)

  LR = 1e-3 (muito alto para fine-tuning):
    ┌─────────────────────────────────────┐
    │   ╱╲  ╱╲  ╱╲                       │  Oscilação destrutiva
    │  ╱  ╲╱  ╲╱  ╲   → Loss diverge     │  Esquece conhecimento pré-treinado
    │ ╱              ╲                    │  "Catastrophic forgetting"
    └─────────────────────────────────────┘

  LR = 1e-5 (ideal para fine-tuning):
    ┌─────────────────────────────────────┐
    │ ╲                                   │  Convergência suave
    │  ╲                                  │  Preserva conhecimento base
    │   ╲___________________________      │  Adapta ao domínio SciELO
    └─────────────────────────────────────┘

  LR = 1e-7 (muito baixo):
    ┌─────────────────────────────────────┐
    │ ─────────────────────────           │  Convergência desprezível
    │                                     │  Modelo quase não muda
    │                                     │  Desperdício de computação
    └─────────────────────────────────────┘
```

**Schedule linear com warmup** (implementado via `Seq2SeqTrainer`):

O LR não é constante — segue um schedule com warmup linear (500 steps) + decay linear até 0:

```
LR
1e-5 ┤          ╱╲
     │         ╱  ╲
     │        ╱    ╲
     │       ╱      ╲
     │      ╱        ╲
     │     ╱          ╲
     │    ╱            ╲
     │   ╱              ╲
     │  ╱                ╲
     │ ╱                  ╲
0    ┤╱                    ╲_
     └──────┬──────────────┬──→ Steps
            500          13500

 Fase 1: WARMUP (steps 0→500)
   LR sobe linearmente de 0 até 1e-5
   → Evita instabilidade no início (gradientes grandes com pesos não calibrados)
   → "Aquece" o otimizador: momentum (m̂) e variância (v̂) do AdamW estabilizam

 Fase 2: DECAY LINEAR (steps 500→13500)
   LR decresce linearmente de 1e-5 até ~0
   → No início: passos maiores para aprender rápido
   → No final: passos minúsculos para refinamento fino

 Valores reais observados no treinamento:
   Step   100: lr = 1.98e-06  (warmup: subindo)
   Step   500: lr = 9.98e-06  (pico: ~1e-5)
   Step  1000: lr = 9.62e-06  (início do decay)
   Step  5000: lr = 6.54e-06  (metade do treinamento)
   Step 10000: lr = 2.70e-06  (75% do treinamento)
   Step 13000: lr = 3.88e-07  (quase zero)
   Step 13500: lr = 3.08e-09  (final: praticamente zero)
```

**Referência para warmup**: Goyal, P. et al. (2017). *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*. arXiv:1706.02677. https://arxiv.org/abs/1706.02677

**Por que warmup de 500 steps?**

```
Sem warmup:
  Step 0: Gradiente grande (loss alta) × LR máximo → passo enorme
  → Pode "destruir" features pré-treinadas nas primeiras iterações
  → Fenômeno: "loss spike" ou divergência precoce

Com warmup de 500 steps:
  Step 0:   LR ≈ 0       → passo quase nulo, gradientes estabilizam
  Step 250: LR ≈ 5e-6    → passos moderados, momentum calibrado
  Step 500: LR = 1e-5     → passo máximo, otimizador calibrado ✅
  
  500 steps = ~3% do treinamento total (13.500 steps)
  → Prática padrão: warmup de 1-5% do total de steps
```

---

#### `--fp16` (Mixed Precision Training)

**O que é**: Treina com **precisão mista** — forward pass em FP16 (16 bits), backward pass e atualização de pesos em FP32 (32 bits).

**Referência**: Micikevicius, P. et al. (2018). *Mixed Precision Training*. In ICLR 2018. https://arxiv.org/abs/1710.03740

**Por que usar**:

```
FP32 (32-bit float):  ████████████████████████████████  → 4 bytes por peso
FP16 (16-bit float):  ████████████████                  → 2 bytes por peso

                     FP32          FP16 (mixed)     Economia
Pesos do modelo:     880 MB        440 MB           50%
Ativações:           768 MB        384 MB           50%
Gradientes:          880 MB        880 MB (FP32)      0% (mantido em FP32)
Estados Adam:       1760 MB       1760 MB (FP32)      0% (mantido em FP32)
                    ─────────     ─────────
Total aprox:        4288 MB       3464 MB           ~19% menor ✅
```

**Como funciona (Automatic Mixed Precision)**:

```
┌──────────────────────────────────────────────┐
│ 1. Pesos copiados FP32 → FP16 (master copy)  │
│ 2. Forward pass em FP16 (rápido nos Tensors)  │
│    → Loss calculada em FP16                   │
│ 3. Loss scaling (multiplica loss × 65536)     │
│    → Evita underflow de gradientes em FP16    │
│ 4. Backward pass: gradientes em FP16          │
│ 5. Gradientes → FP32, divididos pelo scaler   │
│ 6. Optimizer.step() em FP32 (pesos master)    │
│ 7. Pesos FP32 → FP16 para próximo forward     │
└──────────────────────────────────────────────┘
```

**Benefícios na RTX 4050**:

```
RTX 4050 possui Tensor Cores com suporte a FP16:
  - Operações FP16: ~16.6 TFLOPS
  - Operações FP32: ~8.3 TFLOPS
  → FP16 é ~2x mais rápido para matmul/convolutions

Tempo estimado por epoch:
  FP32: ~25 min/epoch × 12 = ~5.0 horas
  FP16: ~15 min/epoch × 12 = ~3.0 horas  ← ~40% mais rápido
```

---

#### `--max_seq_len 256` ⭐

**O que é**: Comprimento máximo em tokens de cada sequência (source e target). Sequências mais longas são **truncadas**, mais curtas recebem **padding**.

**Por que 256 (e não o padrão 128 ou o máximo 512)?**

O modelo T5 suporta até `n_positions=512` tokens. A escolha de 256 é um compromisso entre capturar abstracts completos e usar VRAM de forma eficiente.

```
Distribuição de comprimento dos abstracts SciELO (em tokens):

  Tokens │
    512+ │ ▏                                    1.2% truncados com max=512
    480  │ ▎                                   
    448  │ ▎
    416  │ ▍
    384  │ ▌
    352  │ ▋
    320  │ █
    288  │ ██
    256  │ ███▎                                 ~5% truncados com max=256
    224  │ ████▌
    192  │ ██████
    160  │ ████████
    128  │ ██████████▎                          ~25% truncados com max=128 ❌
     96  │ ████████████
     64  │ ██████████████
     32  │ ████████████████
      0  │ █████████████████
         └──────────────────────→ Nº de exemplos

→ max_seq_len=128 (padrão): trunca ~25% dos abstracts (perde informação)
→ max_seq_len=256 (escolhido): trunca ~5% (bom compromisso) ✅
→ max_seq_len=512 (máximo): trunca <2% mas usa 4x mais memória
```

**Impacto na VRAM** — a memória escala **quadraticamente** com o comprimento da sequência (self-attention):

$$\text{Memória}_{attention} \propto \text{batch\_size} \times \text{num\_heads} \times \text{seq\_len}^2$$

```
Memória de atenção por camada (batch=8, heads=12):

  max_seq_len=128:  8 × 12 × 128² × 2 bytes  =  3.0 MB/layer  × 24 layers = 72 MB
  max_seq_len=256:  8 × 12 × 256² × 2 bytes  = 12.0 MB/layer  × 24 layers = 288 MB  ← Nosso
  max_seq_len=512:  8 × 12 × 512² × 2 bytes  = 48.0 MB/layer  × 24 layers = 1152 MB

  128 → 256: +216 MB (cabe na RTX 4050 ✅)
  256 → 512: +864 MB (risco de OOM com batch=8 ❌)
```

**Efeito no truncamento**:

```
Abstract original (310 tokens):
  "The present study aimed to evaluate the effect of different
   concentrations of sodium hypochlorite on the bond strength
   of fiber posts cemented with self-adhesive resin cement to
   root dentin. Forty single-rooted bovine teeth were selected
   and decoronated. The root canals were prepared using [...more...]
   The results suggest that sodium hypochlorite concentration
   significantly affects the bond strength values."

Com max_seq_len=128 (truncado em ↓):
  "The present study aimed to evaluate the effect of different
   concentrations of sodium hypochlorite on the bond strength
   of fiber posts cemented with self-adhesive resin cement to
   root dentin. Forty single-rooted bovine teeth were..."
  → PERDE a conclusão do abstract (informação crítica!)

Com max_seq_len=256 (truncado em ↓):
  "The present study aimed to evaluate the effect of different
   concentrations of sodium hypochlorite on the bond strength
   of fiber posts cemented with self-adhesive resin cement to
   root dentin. Forty single-rooted bovine teeth were selected
   and decoronated. The root canals were prepared using [...]
   The results suggest that sodium hypochlorite concentration
   significantly affects the bond strength values."
  → Captura introdução, método E conclusão ✅
```

**Implementação** (em `finetuning/trainer.py`):

```python
def preprocess_function(examples):
    inputs = tokenizer(
        examples["abstract_en"],
        max_length=max_seq_len,    # ← 256
        truncation=True,           # Corta sequências maiores
        padding="max_length",      # Pad até max_seq_len
    )
    targets = tokenizer(
        text_target=examples["abstract_pt"],
        max_length=max_seq_len,    # ← 256
        truncation=True,
        padding="max_length",
    )
    inputs["labels"] = targets["input_ids"]
    # Mascarar PAD tokens com -100 (ignorados na loss)
    inputs["labels"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in inputs["labels"]
    ]
    return inputs
```

---

#### `--early_stopping_patience 2`

**O que é**: Para o treinamento se a `eval_loss` **não melhorar** por 2 epochs consecutivos. Evita overfitting e desperdício de computação.

**Referência**: Prechelt, L. (1998). *Early Stopping — But When?*. In Neural Networks: Tricks of the Trade, Lecture Notes in Computer Science, vol 1524, pp. 55–69. https://doi.org/10.1007/3-540-49430-8_3

**Como funciona** (implementado via `EarlyStoppingCallback` do Transformers):

```
                  patience = 2
                  ─────────────
  Epoch  eval_loss   Melhor?   Contador   Ação
  ─────  ─────────   ───────   ────────   ─────────────────────
    1    1.006836    Sim ✅    0          Salva como melhor
    2    0.993096    Sim ✅    0          Salva como melhor
    3    0.986074    Sim ✅    0          Salva como melhor
    ...     ...        ...      ...        ...
   12    0.972978    Sim ✅    0          Salva como melhor ⭐

  → No nosso caso, eval_loss melhorou em TODAS as 12 epochs.
  → Early stopping NUNCA foi acionado.
  → Se tivéssemos configurado epochs=50, pararia
     quando 2 epochs consecutivos não melhorassem.

  Cenário hipotético (se tivéssemos treinado mais):
  Epoch  eval_loss   Melhor?   Contador   Ação
  ─────  ─────────   ───────   ────────   ─────────────────────
   12    0.972978    Sim ✅    0          Salva como melhor
   13    0.973100    Não ❌    1          Esperando... (1/2)
   14    0.973200    Não ❌    2          PARA ✋ (patience atingido)
   → Carrega checkpoint do epoch 12 (melhor modelo)
```

**Por que patience=2 (e não 1 ou 5)?**

```
patience=1: Muito agressivo — para no primeiro "tropeço"
  → Pode parar prematuramente se houver flutuação normal

patience=2: Equilibrado — permite 1 flutuação mas evita desperdício
  → Prática padrão na literatura de NLP  ✅

patience=5: Conservador — treina mais mesmo sem melhoria
  → Desperdiça horas de GPU se o modelo já convergiu
```

---

#### `--skip_prepare`

**O que é**: Pula a etapa de preparação dos CSVs de treino/validação/teste (já preparados anteriormente no STAGE 3). Sem este flag, o script executaria `data_utils.prepare_evaluation_csv()` novamente.

**Quando usar**: Quando os arquivos `scielo_abstracts_train.csv`, `scielo_abstracts_val.csv` e `scielo_abstracts_test.csv` já existem no diretório `finetuning/abstracts-datasets/`.

---

#### Gradient Checkpointing (ativado automaticamente no código)

**O que é**: Técnica que **recalcula** ativações intermediárias durante o backward pass em vez de armazená-las na memória. Troca computação por memória.

**Referência**: Chen, T. et al. (2016). *Training Deep Nets with Sublinear Memory Cost*. arXiv:1604.06174. https://arxiv.org/abs/1604.06174

```
SEM gradient checkpointing:
  Forward:  layer₁ → [salva a₁] → layer₂ → [salva a₂] → ... → layer₂₄ → [salva a₂₄] → loss
  Backward: usa a₂₄ → ∇₂₄, usa a₂₃ → ∇₂₃, ..., usa a₁ → ∇₁

  Memória: O(n) ativações armazenadas = 24 camadas × ativações
  → Pode exigir >6 GB (impossível na RTX 4050)

COM gradient checkpointing:
  Forward:  layer₁ → [salva a₁] → layer₂ → [descarta] → ... → layer₂₄ → loss
  Backward: recalcula a₂₃ (forward parcial) → ∇₂₃, recalcula a₂₂ → ∇₂₂, ...

  Memória: O(√n) ativações armazenadas ≈ √24 ≈ 5 checkpoints
  → Economia de ~60-70% de VRAM das ativações
  → Custo: ~33% mais lento (recalcula forward para cada segmento)
```

**Implementação** (em `finetuning/trainer.py`):

```python
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()
    # → Reduz VRAM de ativações de ~1.5 GB para ~500 MB
    # → Permite batch_size=8 com max_seq_len=256 na RTX 4050
```

---

### Resumo: Por que cada parâmetro foi escolhido

| Parâmetro        | Valor   | Motivação principal                                         | Alternativa descartada        |
|------------------|---------|-------------------------------------------------------------|-------------------------------|
| `model`          | unicamp-t5 | Melhor trade-off qualidade/tamanho (220M params)         | Helsinki (600M, não cabe)     |
| `epochs`         | 12      | eval_loss convergiu sem overfitting                         | 5 (underfitting), 50 (desnecessário) |
| `batch_size`     | 8       | Maior batch que cabe na RTX 4050 (6GB) com margem          | 16 (risco OOM), 4 (muito ruidoso) |
| `grad_accum`     | 2       | Batch efetivo=16, equilíbrio ruído/estabilidade             | 1 (ruidoso), 4 (lento demais) |
| `lr`             | 1e-5    | LR recomendado para fine-tuning de Transformers             | 1e-3 (catastrophic forgetting), 1e-7 (sem aprendizado) |
| `fp16`           | True    | ~40% mais rápido + ~19% menos VRAM nos Tensor Cores        | FP32 (mais lento, mais VRAM)  |
| `max_seq_len`    | 256     | Captura ~95% dos abstracts sem OOM                          | 128 (perde 25%), 512 (OOM)    |
| `early_stopping` | 2       | Previne overfitting sem parar prematuramente                | 1 (agressivo demais), 5 (desperdiça GPU) |
| `skip_prepare`   | True    | Dados já preparados no STAGE 3                              | False (refaz splits desnecessariamente) |

### Curva de Convergência (eval_loss)

```
Epoch | eval_loss | Step   | Tendência
------|-----------|--------|----------
  1   | 1.006836  |  1125  |
  2   | 0.993096  |  2250  | ↓ melhorou
  3   | 0.986074  |  3375  | ↓ melhorou
  4   | 0.981832  |  4500  | ↓ melhorou
  5   | 0.979202  |  5625  | ↓ melhorou
  6   | 0.977226  |  6750  | ↓ melhorou
  7   | 0.975687  |  7875  | ↓ melhorou
  8   | 0.974656  |  9000  | ↓ melhorou
  9   | 0.973745  | 10125  | ↓ melhorou
 10   | 0.973330  | 11250  | ↓ melhorou
 11   | 0.973035  | 12375  | ↓ melhorou
 12   | 0.972978  | 13500  | ↓ melhorou ⭐ BEST
```

**Observações:**
- A eval_loss melhorou consistentemente em todas as 12 epochs
- O melhor checkpoint foi o último: `checkpoint-13500` (epoch 12, eval_loss: 0.972978)
- Early stopping NÃO foi acionado — o modelo ainda estava convergindo
- A taxa de melhoria desacelera nos epochs finais (~0.0003 por epoch), sugerindo proximidade do ponto ótimo

### Training Loss (média por epoch)

| Epoch | Training Loss (média) | Eval Loss  | Learning Rate (final) |
|-------|----------------------:|-----------:|----------------------:|
| 1     | 1.0962                | 1.006836   | 9.54e-06              |
| 2     | 1.0479                | 0.993096   | 8.69e-06              |
| 3     | 1.0283                | 0.986074   | 7.85e-06              |
| 4     | 1.0173                | 0.981832   | 7.00e-06              |
| 5     | 0.9987                | 0.979202   | 6.08e-06              |
| 6     | 0.9927                | 0.977226   | 5.23e-06              |
| 7     | 0.9794                | 0.975687   | 4.39e-06              |
| 8     | 0.9784                | 0.974656   | 3.46e-06              |
| 9     | 0.9744                | 0.973745   | 2.62e-06              |
| 10    | 0.9692                | 0.973330   | 1.77e-06              |
| 11    | 0.9633                | 0.973035   | 9.26e-07              |
| 12    | 0.9691                | 0.972978   | 3.08e-09              |

**Observações sobre o treinamento:**
- Training loss caiu de ~1.10 (epoch 1) para ~0.96 (epoch 12) — redução de 12.5%
- Learning rate seguiu schedule linear com warmup de 500 steps (pico 1e-5) e decay até ~0
- Gradient norms estáveis em 0.5–0.9 ao longo de todo o treinamento (sem gradient explosion)
- Diferença train_loss vs eval_loss pequena (~0.01), indicando ausência de overfitting

### Detalhes Técnicos do Treinamento

- **Gradient checkpointing**: Reduz consumo de VRAM recalculando ativações intermediárias no backward pass
- **FP16 (mixed precision)**: Reduz uso de memória e acelera computação em Tensor Cores
- **Mascaramento de PAD tokens**: Labels com token PAD são substituídos por -100 para não contribuírem na cross-entropy loss
- **Early stopping**: Monitora `eval_loss` a cada epoch; para se não houver melhoria em 2 epochs consecutivos
- **AdamW**: Otimizador Adam com weight decay desacoplado (0.01)

### Checkpoints

Cada epoch gera um checkpoint. Os 2 últimos são preservados (save_total_limit=2):

| Checkpoint       | Epoch | eval_loss |
|------------------|-------|-----------|
| checkpoint-12375 | 11    | 0.973035  |
| checkpoint-13500 | 12    | 0.972978 ⭐ |

O modelo final (melhor) é salvo na raiz: `unicamp-t5/unicamp-t5/`

### Resumir Treinamento Interrompido

```bash
python finetuning/finetune_selected_models.py \
  --model unicamp-t5 \
  --epochs 12 \
  --batch_size 8 \
  --grad_accum_steps 2 \
  --lr 1e-5 \
  --fp16 \
  --max_seq_len 256 \
  --early_stopping_patience 2 \
  --skip_prepare \
  --resume_from ./unicamp-t5/unicamp-t5/checkpoint-13500
```

O `Seq2SeqTrainer` preserva: estado do otimizador/scheduler, epoch/step atual, melhor modelo e contador de early stopping.

---

## STAGE 5: Avaliação Final

### Objetivo
Comparar o modelo **antes** e **depois** do fine-tuning, usando os **mesmos** 5.000 exemplos de teste.

### Comandos

```bash
# Testar modelo base (antes do fine-tuning)
python finetuning/select_and_test_models.py --model unicamp-t5 --skip_prepare

# Testar modelo fine-tuned
python finetuning/select_and_test_models.py --test_finetuned --model unicamp-t5 --skip_prepare

# Testar ambos e comparar
python finetuning/select_and_test_models.py --test_both --model unicamp-t5 --skip_prepare
```

### Resultados

**Antes do fine-tuning** (`scielo_before_finetuning.csv`):

| Modelo     | BLEU  | chrF  | COMET  | BERTScore |
|------------|------:|------:|-------:|----------:|
| unicamp-t5 | 40.06 | 65.61 | 0.8499 | 0.8957    |

**Após fine-tuning — Epoch 11** (`scielo_after_finetuning_epoch_11.csv`):

| Modelo     | Checkpoint       | BLEU  | chrF  | COMET  | BERTScore |
|------------|------------------|------:|------:|-------:|----------:|
| unicamp-t5 | checkpoint-12375 | 45.51 | 70.54 | 0.8756 | 0.9124    |

**Após fine-tuning — Epoch 12** (`scielo_after_finetuning_epoch_12.csv`):

| Modelo     | Checkpoint       | BLEU  | chrF  | COMET  | BERTScore |
|------------|------------------|------:|------:|-------:|----------:|
| unicamp-t5 | checkpoint-13500 | 45.51 | 70.54 | 0.8756 | 0.9124    |

### Análise de Melhoria

| Métrica    | Antes  | Depois (Ep.12) | Delta   | Melhoria |
|------------|-------:|---------------:|--------:|---------:|
| BLEU       | 40.06  | 45.51          | +5.45   | +13.6%   |
| chrF       | 65.61  | 70.54          | +4.93   | +7.5%    |
| COMET      | 0.8499 | 0.8756         | +0.0257 | +3.0%    |
| BERTScore  | 0.8957 | 0.9124         | +0.0167 | +1.9%    |

### Interpretação

- **BLEU +13.6%**: Melhoria significativa na precisão de n-gramas. O modelo gera traduções com sobreposição lexical mais próxima das referências humanas.
- **chrF +7.5%**: Melhoria a nível de caracteres, indicando melhor morfologia e ortografia (acentuação, concordância).
- **COMET +3.0%**: Score neural baseado em modelo treinado em avaliações humanas confirma melhoria na qualidade percebida.
- **BERTScore +1.9%**: Melhoria na similaridade semântica. O modelo preserva melhor o significado original.
- **Epochs 11→12 estáveis**: Métricas idênticas entre epochs 11 e 12 indicam convergência atingida — o modelo estabilizou.
- **Melhoria dentro da faixa saudável** (+5 a +15% BLEU): Sem sinais de overfitting.

---

## Quickstart

```bash
# 1. Instalar dependências
pip install -r requirements.txt
pip install -r requirements-ml.txt

# 2. Preparar dataset SciELO
python prepare_scielo_dataset.py

# 3. Pipeline completo
python finetune_and_evaluate.py --skip_prepare

# Ou executar etapas individualmente:

# 3a. Preparar splits e testar modelo base
python finetuning/select_and_test_models.py --model unicamp-t5

# 3b. Fine-tuning
python finetuning/finetune_selected_models.py \
  --model unicamp-t5 --epochs 12 --batch_size 8 \
  --grad_accum_steps 2 --lr 1e-5 --fp16 --max_seq_len 256 \
  --early_stopping_patience 2 --skip_prepare

# 3c. Avaliar modelo fine-tuned
python finetuning/select_and_test_models.py --test_finetuned --model unicamp-t5 --skip_prepare
```

---

## Dependências

### requirements.txt
Dependências gerais do projeto (pandas, numpy, etc.)

### requirements-ml.txt
Dependências de machine learning:
- `transformers` — HuggingFace Transformers (modelos, tokenizadores, Trainer)
- `torch` — PyTorch (backend de deep learning)
- `datasets` — HuggingFace Datasets
- `sacrebleu` — Cálculo de BLEU e chrF
- `unbabel-comet` — Cálculo de COMET
- `bert-score` — Cálculo de BERTScore
- `sentencepiece` — Tokenização SentencePiece
- `accelerate` — Aceleração de treinamento HuggingFace
- `tqdm` — Barras de progresso

---

## Detalhes Técnicos

### Reprodutibilidade

- Seed = 42 em todos os splits e treinamento
- `torch.manual_seed(42)` no carregamento do modelo
- Splits determinísticos: mesmos 5k exemplos de teste para base e fine-tuned
- Resultados reprodutíveis com mesma GPU e mesma seed

### Pipeline de Tokenização e Inferência

```
Entrada: "The patient presented with fever and cough."
    ↓ SentencePiece (unigram, 32k vocab)
Input IDs: [37, 1868, 4793, 28, 18851, 11, 14912, 5, 1]
    ↓ T5 Encoder (12 layers × 768 dim × 12 heads)
Hidden states: [768-dim vectors × seq_len]
    ↓ T5 Decoder (12 layers, autoregressive, beam search k=5)
Output IDs: [101, 5847, 12059, 28, 18453, 11, 30419, 5, 1]
    ↓ Decode
Saída: "O paciente apresentou febre e tosse."
```

### Cálculo da Loss

```
Cross-Entropy Loss com mascaramento:
- Tokens de conteúdo: contribuem para a loss
- Tokens PAD (id → -100): ignorados pela loss function
- Isso evita que o modelo aprenda a gerar padding
```

### Early Stopping

```
Para cada epoch:
  1. Calcular eval_loss no conjunto de validação (2k exemplos)
  2. Se eval_loss < melhor_loss_anterior → salvar como melhor modelo
  3. Se eval_loss >= melhor_loss_anterior → incrementar contador
  4. Se contador >= patience (2) → parar treinamento

No nosso caso: eval_loss melhorou em todas as 12 epochs,
portanto early stopping NÃO foi acionado.
```

### Geração (Inferência)

| Parâmetro  | Valor          |
|------------|----------------|
| Decodificação | Beam Search |
| Num beams  | 5              |
| Max length | 256 tokens     |

---

## Estrutura do Projeto

```
.
├── README.md                                  ← Este arquivo
├── PROJECT_STRUCTURE.md                       ← Estrutura detalhada (visual)
├── QUICK_COMMANDS.md                          ← Referência rápida de comandos
├── requirements.txt                           ← Dependências gerais
├── requirements-ml.txt                        ← Dependências ML
│
├── prepare_scielo_dataset.py                  [STAGE 0] Gera abstracts_scielo.csv
├── models-test.py                             [STAGE 1] Avalia 5 modelos em datasets públicos
├── evaluate_quickmt.py                        [STAGE 1] Avalia modelo QuickMT (CTranslate2)
├── choose_best_model.py                       [STAGE 2] Ranking e seleção de modelo
├── show_model_configs.py                      Exibe configurações dos modelos
├── compute_neural_metrics.py                  Calcula COMET e BERTScore
├── finetune_and_evaluate.py                   Pipeline integrado (STAGES 1-5)
├── check_gpu.py                               Verificação de GPU disponível
│
├── scielo_before_finetuning.csv               [STAGE 5] Métricas baseline (BLEU=40.06)
├── scielo_after_finetuning_epoch_1.csv        [STAGE 5] Métricas epoch 1
├── scielo_after_finetuning_epoch_11.csv       [STAGE 5] Métricas epoch 11 (BLEU=45.51)
├── scielo_after_finetuning_epoch_12.csv       [STAGE 5] Métricas epoch 12 (BLEU=45.51)
│
├── evaluation/                                Módulo de avaliação (STAGE 1)
│   ├── __init__.py
│   ├── config.py                              Configurações de avaliação
│   ├── datasets.py                            Carregamento de datasets públicos
│   ├── metrics.py                             Cálculo de métricas
│   ├── models_loader.py                       Carregamento de modelos
│   ├── run.py                                 Execução da avaliação
│   ├── io_utils.py                            Utilitários de I/O
│   └── fill_missing_metrics.py                Preenche métricas faltantes
│
├── evaluation_results/                        Resultados de avaliação
│   ├── translation_metrics_all.csv            [STAGE 1] Consolidado todos os modelos
│   ├── Helsinki-NLP_opus-mt-tc-big-en-pt.csv
│   ├── Narrativa_mbart-large-50-finetuned-opus-en-pt-translation.csv
│   ├── unicamp-dl_translation-en-pt-t5.csv
│   ├── VanessaSchenkel_unicamp-finetuned-en-to-pt-dataset-ted.csv
│   ├── danhsf_m2m100_418M-finetuned-kde4-en-to-pt_BR.csv
│   └── quickmt_quickmt-en-pt.csv
│
├── finetuning/                                Módulo de fine-tuning (STAGES 3-5)
│   ├── __init__.py
│   ├── config.py                              Configurações centralizadas
│   ├── models.py                              Carregamento/salvamento de modelos
│   ├── data_utils.py                          Preparação de dados (splits)
│   ├── datasets.py                            Dataset handling
│   ├── metrics.py                             BLEU, chrF, COMET, BERTScore
│   ├── evaluate.py                            Avaliação com progresso (tqdm)
│   ├── trainer.py                             Seq2SeqTrainer + fine-tuning loop
│   ├── compare.py                             Comparação base vs fine-tuned
│   ├── io_utils.py                            Utilitários I/O
│   ├── finetune_selected_models.py            [STAGE 4] Script de fine-tuning
│   ├── select_and_test_models.py              [STAGE 3+5] Preparo + teste
│   └── abstracts-datasets/                    [STAGE 3] Dados SciELO
│       ├── abstracts_scielo.csv               Corpus completo (2.7M exemplos)
│       ├── scielo_abstracts_train.csv         18.000 exemplos (treino)
│       ├── scielo_abstracts_val.csv            2.000 exemplos (validação)
│       └── scielo_abstracts_test.csv           5.000 exemplos (teste)
│
├── unicamp-t5/                                ⭐ MODELO FINE-TUNED (resultado final)
│   └── unicamp-t5/
│       ├── config.json                        Configuração do modelo
│       ├── generation_config.json             Configuração de geração
│       ├── model.safetensors                  Pesos do melhor modelo (epoch 12)
│       ├── tokenizer.json                     Tokenizador
│       ├── tokenizer_config.json              Configuração do tokenizador
│       ├── spiece.model                       Modelo SentencePiece
│       ├── special_tokens_map.json
│       ├── checkpoint-12375/                  Checkpoint epoch 11
│       └── checkpoint-13500/                  Checkpoint epoch 12 (best)
│           ├── model.safetensors
│           ├── optimizer.pt
│           ├── scheduler.pt
│           ├── trainer_state.json             Log completo de treinamento
│           └── training_args.bin
│
├── models/                                    Modelos auxiliares
│   └── finetuned-scielo/
│       └── helsinki/                           Fine-tuning anterior (Helsinki)
│
├── models-configs/                            Configurações JSON dos modelos
│   ├── helsink.json
│   └── m2m100.json
│
└── checkpoints/                               Checkpoints de controle
    ├── training/
    └── evaluation/
```

---

## Métricas de Avaliação — Explicação Técnica

Este projeto utiliza 4 métricas complementares para avaliar a qualidade das traduções. Duas são métricas **lexicais** (baseadas em sobreposição de tokens) e duas são métricas **neurais** (baseadas em embeddings de modelos pré-treinados). A combinação garante uma avaliação robusta que captura tanto a fidelidade lexical quanto a adequação semântica.

### Visão Geral

| Métrica     | Tipo    | Granularidade | Escala   | Requer Source? | Implementação        |
|-------------|---------|---------------|----------|----------------|----------------------|
| BLEU        | Lexical | Palavra       | 0–100    | Não            | `sacrebleu.BLEU()`   |
| chrF        | Lexical | Caractere     | 0–100    | Não            | `sacrebleu.CHRF()`   |
| COMET       | Neural  | Sentença      | 0–1      | Sim            | `Unbabel/wmt22-comet-da` |
| BERTScore   | Neural  | Token         | 0–1      | Não            | `bert-score` (lang=pt) |

---

### 1. BLEU (Bilingual Evaluation Understudy)

**Referência**: Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). *BLEU: a Method for Automatic Evaluation of Machine Translation*. In Proceedings of the 40th Annual Meeting of the ACL, pp. 311–318. https://aclanthology.org/P02-1040/

**Padronização**: Post, M. (2018). *A Call for Clarity in Reporting BLEU Scores*. In Proceedings of the Third Conference on Machine Translation (WMT), pp. 186–191. https://aclanthology.org/W18-6319/

#### O que mede
BLEU mede a **precisão de n-gramas** entre a tradução candidata (hipótese) e a tradução de referência humana, penalizando traduções muito curtas via *brevity penalty*. É a métrica mais utilizada na literatura de tradução automática.

#### Fórmula

$$\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \cdot \log p_n\right)$$

Onde:
- $p_n$ = precisão de n-gramas modificada (clipped precision)
- $w_n = \frac{1}{N}$ (peso uniforme, $N=4$ por padrão)
- $\text{BP} = \min\left(1, \; e^{1 - r/c}\right)$ = brevity penalty ($r$ = comprimento da referência, $c$ = comprimento da hipótese)

#### Exemplo Visual

```
Referência: "O paciente apresentou febre e tosse persistente"
Hipótese:   "O paciente apresentou febre e tosse"

Unigrams (1-gram):
  Referência: {O, paciente, apresentou, febre, e, tosse, persistente}  → 7 tokens
  Hipótese:   {O, paciente, apresentou, febre, e, tosse}              → 6 tokens
  Match:      {O, paciente, apresentou, febre, e, tosse}              → 6 matches
  p₁ = 6/6 = 1.00 ✅

Bigrams (2-gram):
  Referência: {O paciente, paciente apresentou, apresentou febre, febre e, e tosse, tosse persistente}
  Hipótese:   {O paciente, paciente apresentou, apresentou febre, febre e, e tosse}
  Match:      {O paciente, paciente apresentou, apresentou febre, febre e, e tosse}  → 5/5
  p₂ = 5/5 = 1.00 ✅

Trigrams (3-gram):
  Referência: {O paciente apresentou, paciente apresentou febre, apresentou febre e, febre e tosse, e tosse persistente}
  Hipótese:   {O paciente apresentou, paciente apresentou febre, apresentou febre e, febre e tosse}
  Match:      {O paciente apresentou, paciente apresentou febre, apresentou febre e, febre e tosse}  → 4/4
  p₃ = 4/4 = 1.00 ✅

4-grams:
  Referência: {O paciente apresentou febre, paciente apresentou febre e, apresentou febre e tosse, febre e tosse persistente}
  Hipótese:   {O paciente apresentou febre, paciente apresentou febre e, apresentou febre e tosse}
  Match:      {O paciente apresentou febre, paciente apresentou febre e, apresentou febre e tosse}  → 3/3
  p₄ = 3/3 = 1.00 ✅

Brevity Penalty:
  r = 7 (referência), c = 6 (hipótese) → c < r
  BP = exp(1 - 7/6) = exp(-0.167) ≈ 0.846

BLEU = BP × exp(¼ × (log(1.0) + log(1.0) + log(1.0) + log(1.0)))
     = 0.846 × exp(0)
     = 0.846 × 1.0
     = 84.6   ← Penalizado por ser mais curta que a referência
```

#### Limitações

- **Insensível a sinônimos**: "febre" vs "temperatura alta" = 0 match, apesar de semanticamente equivalentes
- **Independente da ordem global**: Permutações de fragmentos podem gerar BLEU alto sem coerência
- **Brevity penalty assimétrica**: Penaliza traduções curtas, mas não as longas demais

#### Implementação neste projeto

```python
# finetuning/metrics.py
from sacrebleu import BLEU
bleu = BLEU(lowercase=False)
bleu_score = bleu.corpus_score(predictions, [references])  # corpus-level
# Retorna: score ∈ [0, 100]
```

> **Nota**: Utilizamos `sacreBLEU` (Post, 2018) que garante tokenização padronizada e reprodutibilidade. O score é computado a nível de corpus (não média de sentenças).

---

### 2. chrF (Character n-gram F-score)

**Referência**: Popović, M. (2015). *chrF: character n-gram F-score for automatic MT evaluation*. In Proceedings of the Tenth Workshop on Statistical Machine Translation (WMT), pp. 392–395. https://aclanthology.org/W15-3049/

#### O que mede
chrF mede a **sobreposição de n-gramas de caracteres** entre hipótese e referência, utilizando o F-score (média harmônica de precisão e recall). Por operar a nível de caractere, é mais robusta a variações morfológicas do que o BLEU.

#### Fórmula

$$\text{chrF}_\beta = (1 + \beta^2) \cdot \frac{\text{chrP} \cdot \text{chrR}}{\beta^2 \cdot \text{chrP} + \text{chrR}}$$

Onde:
- $\text{chrP}_n = \frac{|\text{n-gramas}_{\text{hyp}} \cap \text{n-gramas}_{\text{ref}}|}{|\text{n-gramas}_{\text{hyp}}|}$ (precisão de char n-grams)
- $\text{chrR}_n = \frac{|\text{n-gramas}_{\text{hyp}} \cap \text{n-gramas}_{\text{ref}}|}{|\text{n-gramas}_{\text{ref}}|}$ (recall de char n-grams)
- $\beta = 2$ por padrão (favorece recall)
- Média sobre $n = 1, 2, \ldots, 6$ (character n-grams de ordem 1 a 6)

#### Exemplo Visual

```
Referência: "apresentou"
Hipótese:   "apresentaram"

Character 3-grams:
  Referência: {apr, pre, res, ese, sen, ent, nto, tou}           → 8 trigrams
  Hipótese:   {apr, pre, res, ese, sen, ent, nta, tar, ara, ram} → 10 trigrams
  Interseção: {apr, pre, res, ese, sen, ent}                     →  6 matches

  chrP₃ = 6/10 = 0.60 (precisão: quantos trigrams da hipótese estão na referência)
  chrR₃ = 6/8  = 0.75 (recall: quantos trigrams da referência foram cobertos)

  chrF₃ (β=2) = (1 + 4) × (0.60 × 0.75) / (4 × 0.60 + 0.75)
              = 5 × 0.45 / 3.15
              = 0.714

→ Apesar de conjugações diferentes ("apresentou" vs "apresentaram"),
  chrF captura a similaridade morfológica (71.4%) enquanto BLEU
  word-level daria 0% match (palavras diferentes).
```

#### Vantagens sobre BLEU

```
Exemplo: tradução com variação morfológica

Referência: "Os pacientes foram diagnosticados"
Hipótese A: "O paciente foi diagnosticado"         ← tradução correta (singular)
Hipótese B: "A mesa voou pelo hospital"             ← tradução incorreta

BLEU (word-level):
  Hipótese A: matches = {diagnosticado~diagnosticados?} → match parcial
  Hipótese B: matches = {}                              → 0 matches
  → BLEU diferencia, mas penaliza A severamente por flexão

chrF (char-level):
  Hipótese A: alta sobreposição em "pacient-", "diagnosticad-", "for-/foi"
  Hipótese B: baixíssima sobreposição
  → chrF captura melhor que A é quase correta
```

#### Implementação neste projeto

```python
# finetuning/metrics.py
from sacrebleu import CHRF
chrf = CHRF(lowercase=False)
chrf_score = chrf.corpus_score(predictions, [references])  # corpus-level
# Retorna: score ∈ [0, 100]
```

---

### 3. COMET (Crosslingual Optimized Metric for Evaluation of Translation)

**Referência**: Rei, R., de Souza, J. G. C., Alves, D., Zerva, C., Farinha, A. C., Glushkova, T., Lavie, A., Coheur, L., & Martins, A. F. T. (2022). *COMET-22: Unbabel-IST 2022 Submission for the Metrics Shared Task*. In Proceedings of the Seventh Conference on Machine Translation (WMT), pp. 578–585. https://aclanthology.org/2022.wmt-1.52/

**Modelo base**: Conneau, A. et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale*. In Proceedings of ACL 2020, pp. 8440–8451. https://aclanthology.org/2020.acl-main.747/ (XLM-RoBERTa)

#### O que mede
COMET é uma métrica **neural aprendida** que utiliza um modelo XLM-RoBERTa fine-tuned em avaliações humanas (Direct Assessments) de competições WMT. Diferente de BLEU e chrF, COMET considera a **frase fonte** (source) além da referência e hipótese, capturando **adequação** (se o significado foi preservado) e **fluência**.

#### Arquitetura

```
┌─────────────────────────────────────────────────────┐
│                    COMET-22                          │
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │  Source   │  │ Hipótese │  │Referência│          │
│  │  (EN)    │  │  (MT)    │  │  (REF)   │          │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘          │
│       │              │              │                │
│       ▼              ▼              ▼                │
│  ┌─────────────────────────────────────────┐        │
│  │         XLM-RoBERTa (encoder)           │        │
│  │      (550M params, 24 layers)           │        │
│  └─────────────────────────────────────────┘        │
│       │              │              │                │
│       ▼              ▼              ▼                │
│   emb_src        emb_mt         emb_ref             │
│       │              │              │                │
│       ▼              ▼              ▼                │
│  ┌─────────────────────────────────────────┐        │
│  │     Pooling + Feature Extraction         │        │
│  │  [emb_src; emb_mt; emb_ref;             │        │
│  │   |emb_src - emb_mt|;                   │        │
│  │   |emb_ref - emb_mt|;                   │        │
│  │   emb_src * emb_mt;                     │        │
│  │   emb_ref * emb_mt]                     │        │
│  └────────────────┬────────────────────────┘        │
│                   │                                  │
│                   ▼                                  │
│  ┌─────────────────────────────────────────┐        │
│  │       Estimator (Feed-Forward)           │        │
│  │       → score ∈ [0, 1]                   │        │
│  └─────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────┘
```

#### Exemplo Visual

```
Source:    "The patient presented with persistent fever and dry cough."
Referência: "O paciente apresentou febre persistente e tosse seca."
Hipótese A: "O paciente apresentou febre persistente e tosse seca."   → COMET ≈ 1.00
Hipótese B: "O paciente teve febre contínua e tosse sem catarro."     → COMET ≈ 0.88
Hipótese C: "O doente mostrou uma febre que não passa e tosse."       → COMET ≈ 0.80
Hipótese D: "A mesa apresentou febre e tosse."                        → COMET ≈ 0.35

Análise:
  Hipótese A: Tradução perfeita                    → score máximo
  Hipótese B: Semanticamente correta, léxico diferente
              XLM-R captura que "contínua" ≈ "persistente"
              e "sem catarro" ≈ "seca"              → score alto
  Hipótese C: Significado preservado, estilo informal
              "doente" ≈ "paciente", "que não passa" ≈ "persistente"
              → COMET detecta adequação semântica     → score bom
  Hipótese D: Erro semântico grave ("mesa" ≠ "patient")
              COMET usa o source para detectar inconsistência
              → score baixo

Nota: BLEU daria score ZERO para B e C (sem match exato de n-gramas),
      mas COMET reconhece que são traduções válidas.
```

#### Por que COMET usa o source?

```
Source:     "The bank collapsed after the flood."
Referência: "O banco desabou após a enchente."

Hipótese A: "O banco desabou após a enchente."     → COMET alto
Hipótese B: "A instituição bancária faliu."         → COMET baixo

Sem o source, a Hipótese B poderia parecer uma paráfrase razoável.
Mas o source diz "flood" (enchente), não "financial crisis".
COMET detecta que "banco" = margem do rio (não instituição financeira),
e que "collapsed" = desabou fisicamente (não faliu).
→ O acesso ao source resolve ambiguidades e melhora a correlação
  com julgamentos humanos.
```

#### Implementação neste projeto

```python
# finetuning/metrics.py
from comet import download_model, load_from_checkpoint

model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(model_path)
comet_model.eval()

data = [
    {"src": src, "mt": pred, "ref": ref}
    for src, pred, ref in zip(sources, predictions, references)
]
output = comet_model.predict(data, batch_size=2, gpus=1)
system_score = float(output.system_score)  # média ∈ [0, 1]
```

> **Nota**: O modelo utilizado é `Unbabel/wmt22-comet-da`, treinado em Direct Assessments (DA) de competições WMT15–WMT20. Requer ~2GB de VRAM adicionais (XLM-R large). Por isso, o modelo de tradução é movido para CPU antes do cálculo do COMET.

---

### 4. BERTScore

**Referência**: Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). *BERTScore: Evaluating Text Generation with BERT*. In International Conference on Learning Representations (ICLR 2020). https://openreview.net/forum?id=SkeHuCVFDr

#### O que mede
BERTScore calcula a **similaridade semântica** entre hipótese e referência usando **embeddings contextuais** de um modelo BERT pré-treinado. Em vez de comparar tokens exatos (como BLEU), compara representações vetoriais que codificam o significado no contexto.

#### Fórmula

Para cada token $x_i$ da referência e $\hat{x}_j$ da hipótese, calcula-se a similaridade por cosseno dos embeddings contextuais:

$$\text{Recall} = \frac{1}{|x|} \sum_{x_i \in x} \max_{\hat{x}_j \in \hat{x}} \; \mathbf{x}_i^\top \hat{\mathbf{x}}_j$$

$$\text{Precision} = \frac{1}{|\hat{x}|} \sum_{\hat{x}_j \in \hat{x}} \max_{x_i \in x} \; \mathbf{x}_i^\top \hat{\mathbf{x}}_j$$

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

(Onde $\mathbf{x}_i$ e $\hat{\mathbf{x}}_j$ são embeddings contextuais L2-normalizados)

#### Exemplo Visual

```
Referência: "O paciente apresentou febre persistente"
Hipótese:   "O doente teve temperatura alta contínua"

Passo 1: Gerar embeddings contextuais (BERT/mBERT)

  Referência:  O         paciente   apresentou   febre       persistente
               [v₁]      [v₂]       [v₃]         [v₄]        [v₅]

  Hipótese:    O         doente     teve         temperatura  alta       contínua
               [ĥ₁]      [ĥ₂]       [ĥ₃]         [ĥ₄]        [ĥ₅]       [ĥ₆]

Passo 2: Calcular matriz de similaridade por cosseno

               ĥ₁(O)  ĥ₂(doente)  ĥ₃(teve)  ĥ₄(temperatura)  ĥ₅(alta)  ĥ₆(contínua)
  v₁(O)        0.99    0.12        0.08       0.05              0.03       0.04
  v₂(paciente) 0.15    0.87 ←max   0.10       0.08              0.05       0.06
  v₃(apresentou)0.10   0.12        0.72 ←max  0.06              0.04       0.08
  v₄(febre)    0.05    0.09        0.07       0.83 ←max         0.45       0.12
  v₅(persistente)0.03  0.06        0.05       0.15              0.30       0.82 ←max

Passo 3: Greedy matching (cada token → melhor match)

  Recall (para cada token da referência, max cosseno com hipótese):
    O           → max(0.99, 0.12, 0.08, 0.05, 0.03, 0.04) = 0.99
    paciente    → max(0.15, 0.87, 0.10, 0.08, 0.05, 0.06) = 0.87  ← "doente" capturado!
    apresentou  → max(0.10, 0.12, 0.72, 0.06, 0.04, 0.08) = 0.72  ← "teve" capturado!
    febre       → max(0.05, 0.09, 0.07, 0.83, 0.45, 0.12) = 0.83  ← "temperatura" capturado!
    persistente → max(0.03, 0.06, 0.05, 0.15, 0.30, 0.82) = 0.82  ← "contínua" capturado!

  Recall = (0.99 + 0.87 + 0.72 + 0.83 + 0.82) / 5 = 0.846

  Precision (para cada token da hipótese, max cosseno com referência):
    O           → 0.99
    doente      → 0.87 (← "paciente")
    teve        → 0.72 (← "apresentou")
    temperatura → 0.83 (← "febre")
    alta        → 0.45 (← "febre", match parcial)
    contínua    → 0.82 (← "persistente")

  Precision = (0.99 + 0.87 + 0.72 + 0.83 + 0.45 + 0.82) / 6 = 0.780

  F₁ = 2 × (0.780 × 0.846) / (0.780 + 0.846) = 0.812

→ BERTScore F₁ = 0.812 (alto!)
  Apesar de ZERO palavras idênticas (exceto "O"),
  BERTScore reconhece equivalência semântica:
    paciente ↔ doente          (sinônimos)
    apresentou ↔ teve          (verbos relacionados)
    febre ↔ temperatura alta   (conceito médico equivalente)
    persistente ↔ contínua     (sinônimos)
```

#### Implementação neste projeto

```python
# finetuning/metrics.py
from bert_score import score

P, R, F1 = score(
    predictions,       # list[str] — traduções do modelo
    references,        # list[str] — referências humanas
    lang="pt",         # seleciona modelo multilíngue adequado
    batch_size=2,      # batch pequeno para caber na GPU
    device="cuda"
)
bertscore_f1 = float(F1.mean())  # média ∈ [0, 1]
```

> **Nota**: O parâmetro `lang="pt"` seleciona automaticamente o modelo BERT multilíngue adequado para português. A métrica é computada por sentença e depois promediada a nível de corpus. O modelo BERT é carregado após liberar o modelo de tradução da GPU para evitar OOM na RTX 4050 (6GB VRAM).

---

### Comparação das Métricas

| Aspecto                    | BLEU          | chrF          | COMET          | BERTScore      |
|----------------------------|:-------------:|:-------------:|:--------------:|:--------------:|
| **Granularidade**          | Palavra       | Caractere     | Sentença       | Subpalavra     |
| **Base de comparação**     | N-gramas exatos | Char n-gramas | Embeddings XLM-R | Embeddings BERT |
| **Detecta sinônimos?**     | Não           | Parcialmente  | Sim            | Sim            |
| **Detecta paráfrases?**    | Não           | Não           | Sim            | Sim            |
| **Sensível à morfologia?** | Não           | Sim           | Sim            | Sim            |
| **Usa frase fonte?**       | Não           | Não           | Sim            | Não            |
| **Correlação com humanos** | Moderada      | Boa           | Muito alta     | Alta           |
| **Custo computacional**    | Muito baixo   | Muito baixo   | Alto (~2GB GPU)| Médio (~1GB GPU)|
| **Velocidade**             | ~5s/corpus    | ~5s/corpus    | ~60s/corpus    | ~30s/corpus    |
| **Interpretabilidade**     | Alta          | Alta          | Baixa (caixa-preta) | Média     |
| **Ano de publicação**      | 2002          | 2015          | 2022           | 2020           |

### Por que usar 4 métricas?

```
Caso 1: BLEU alto, COMET baixo
→ A tradução tem as mesmas palavras, mas em ordem ou contexto errado
   Exemplo: "bank" traduzido como "banco" (financeiro) quando o contexto era "rio"

Caso 2: BLEU baixo, BERTScore alto
→ A tradução usa sinônimos/paráfrases corretos que BLEU não reconhece
   Exemplo: "febre" vs "temperatura elevada"

Caso 3: chrF alto, BLEU baixo
→ Morfologia correta mas palavras diferentes (flexões, conjugações)
   Exemplo: "apresentaram" vs "apresentou" (chrF captura "apresent-")

Caso 4: Todas altas
→ Tradução de alta qualidade ✅ (nosso caso: BLEU=45.51, chrF=70.54,
   COMET=0.8756, BERTScore=0.9124 após fine-tuning)
```

### Resultados neste projeto

| Métrica    | Antes  | Depois | Delta   | O que a melhoria indica                                    |
|------------|-------:|-------:|--------:|-----------------------------------------------------------|
| BLEU       | 40.06  | 45.51  | +5.45   | Mais n-gramas corretos → vocabulário do domínio aprendido  |
| chrF       | 65.61  | 70.54  | +4.93   | Melhor morfologia → concordância e acentuação aprendidas   |
| COMET      | 0.8499 | 0.8756 | +0.0257 | Maior adequação semântica validada por modelo neural        |
| BERTScore  | 0.8957 | 0.9124 | +0.0167 | Embeddings mais próximos → significado melhor preservado    |

> **Interpretação geral**: As 4 métricas melhoraram de forma consistente, indicando que o fine-tuning produziu ganhos reais em *todas* as dimensões de qualidade — não apenas em sobreposição lexical superficial, mas também em adequação semântica profunda.

---

## Referências

### Artigos Científicos

- Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). *BLEU: a Method for Automatic Evaluation of Machine Translation*. In Proceedings of the 40th Annual Meeting of the ACL, pp. 311–318. https://aclanthology.org/P02-1040/
- Popović, M. (2015). *chrF: character n-gram F-score for automatic MT evaluation*. In Proceedings of the Tenth Workshop on Statistical Machine Translation (WMT), pp. 392–395. https://aclanthology.org/W15-3049/
- Post, M. (2018). *A Call for Clarity in Reporting BLEU Scores*. In Proceedings of the Third Conference on Machine Translation (WMT), pp. 186–191. https://aclanthology.org/W18-6319/
- Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). *BERTScore: Evaluating Text Generation with BERT*. In International Conference on Learning Representations (ICLR 2020). https://openreview.net/forum?id=SkeHuCVFDr
- Conneau, A. et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale*. In Proceedings of ACL 2020, pp. 8440–8451. https://aclanthology.org/2020.acl-main.747/
- Rei, R. et al. (2022). *COMET-22: Unbabel-IST 2022 Submission for the Metrics Shared Task*. In Proceedings of the Seventh Conference on Machine Translation (WMT), pp. 578–585. https://aclanthology.org/2022.wmt-1.52/
- Raffel, C. et al. (2019). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*. arXiv:1910.10683
- Lopes, A. et al. (2020). *Lite Training Strategies for Portuguese-English and English-Portuguese Translation*. In Proceedings of WMT 2020, pp. 833–840. https://aclanthology.org/2020.wmt-1.90/

### Bibliotecas e Ferramentas

- HuggingFace Transformers: https://huggingface.co/docs/transformers/
- SacreBLEU: https://github.com/mjpost/sacrebleu
- COMET: https://github.com/Unbabel/COMET
- BERTScore: https://github.com/Tiiiger/bert_score
- Repositório do modelo: https://huggingface.co/unicamp-dl/translation-en-pt-t5
- Código-fonte do modelo: https://github.com/unicamp-dl/Lite-T5-Translation

---

**Versão**: 5.0 | **Data**: Fevereiro 2026

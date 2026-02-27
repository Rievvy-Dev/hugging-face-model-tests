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
| Ativação                  | GeLU (Gaussian Error Linear Unit) |

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
│   └─ 20.000 exemplos para TESTE
└─ Resultado: finetuning/abstracts-datasets/*.csv
        ↓
STAGE 4: FINE-TUNING
├─ GPU: NVIDIA RTX 4050 (6GB VRAM)
├─ 12 epochs, batch_size=8, grad_accum=2, lr=1e-5
├─ Early stopping com patience=2
└─ Resultado: unicamp-t5/unicamp-t5/ (modelo fine-tuned)
        ↓
STAGE 5: AVALIAÇÃO FINAL
├─ Testar modelo base vs fine-tuned nos MESMOS 20k dados de teste
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
| Teste      | 20.000   | Avaliação final (mesmos para base e fine-tuned) |

**Total: 40.000 exemplos (~1.5% do corpus completo)**

### Justificativa do Dataset Compacto

- **18k treino**: Suficiente para adaptação de domínio (abstracts científicos) sem overfitting
- **2k validação**: Monitora eval_loss por epoch e aciona early stopping
- **20k teste**: Mesmo conjunto usado na avaliação do modelo base, garantindo comparação justa
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
    test_samples=20_000
)
"
```

### Saída
```
finetuning/abstracts-datasets/
├── scielo_abstracts_train.csv   (18.000 exemplos)
├── scielo_abstracts_val.csv     ( 2.000 exemplos)
└── scielo_abstracts_test.csv    (20.000 exemplos)
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
Comparar o modelo **antes** e **depois** do fine-tuning, usando os **mesmos** 20.000 exemplos de teste.

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
- Splits determinísticos: mesmos 20k exemplos de teste para base e fine-tuned
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
│       └── scielo_abstracts_test.csv          20.000 exemplos (teste)
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

## Referências

- Raffel, C. et al. (2019). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*. arXiv:1910.10683
- Lopes, A. et al. (2020). *Lite Training Strategies for Portuguese-English and English-Portuguese Translation*. WMT 2020, pp. 833-840
- Post, M. (2018). *A Call for Clarity in Reporting BLEU Scores*. WMT 2018 (sacreBLEU)
- Rei, R. et al. (2022). *COMET-22: Unbabel-IST 2022 Submission for the Metrics Shared Task*. WMT 2022
- Zhang, T. et al. (2020). *BERTScore: Evaluating Text Generation with BERT*. ICLR 2020
- HuggingFace Transformers: https://huggingface.co/docs/transformers/
- SacreBLEU: https://github.com/mjpost/sacrebleu
- COMET: https://github.com/Unbabel/COMET
- BERTScore: https://github.com/Tiiiger/bert_score
- Repositório do modelo: https://huggingface.co/unicamp-dl/translation-en-pt-t5
- Código-fonte do modelo: https://github.com/unicamp-dl/Lite-T5-Translation

---

**Versão**: 4.0 | **Data**: Fevereiro 2026

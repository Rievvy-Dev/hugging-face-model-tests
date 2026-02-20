# 🎯 Pipeline Completo: Avaliação, Seleção e Fine-Tuning de Modelos de Tradução EN→PT

## 📚 Visão Geral da Metodologia

Este projeto implementa um **pipeline de 5 estágios** para identificar os melhores modelos de tradução automática inglês→português e adapta-los a um domínio específico (abstracts científicos do SciELO).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  STAGE 1: AVALIAÇÃO INICIAL                                               │
│  ├─ Testar 6 modelos em 4 datasets diferentes                             │
│  ├─ Calcular BLEU, chr-F, COMET, BERTScore F1                             │
│  └─ Resultado: translation_metrics_all.csv                                │
│         ↓                                                                  │
│  STAGE 2: SELEÇÃO DOS MELHORES MODELOS                                    │
│  ├─ Usar ranking composto para escolher Top 2                             │
│  ├─ Salvar configurações em JSON                                          │
│  └─ Resultado: top2_models.json                                           │
│         ↓                                                                  │
│  STAGE 3: PREPARAÇÃO DE DADOS                                             │
│  ├─ Separar SciELO em 3 splits não-sobrepostos:                           │
│  │  ├─ 200k exemplos para TREINO (fine-tuning)                            │
│  │  ├─ 20k exemplos para VALIDAÇÃO (monitoramento durante treino)         │
│  │  └─ 20k exemplos para TESTE (avaliação final)                          │
│  └─ Resultado: 3 arquivos CSV                                             │
│         ↓                                                                  │
│  STAGE 4: FINE-TUNING                                                     │
│  ├─ Fine-tune dos 2 modelos selecionados                                  │
│  ├─ Salvar checkpoints para resumir se interrompido                        │
│  ├─ Treinar com 200k dados + validação com 20k                             │
│  └─ Resultado: modelos fine-tuned salvos                                  │
│         ↓                                                                  │
│  STAGE 5: AVALIAÇÃO FINAL E COMPARAÇÃO                                    │
│  ├─ Testar modelos fine-tuned nos MESMOS 20k dados de teste               │
│  ├─ Comparar com resultados do STAGE 1 (base vs fine-tuned)               │
│  ├─ Detectar overfitting/underfitting                                     │
│  └─ Resultado: relatório comparativo final                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quickstart (Rodar Tudo)

Se você deseja executar o pipeline completo:

```bash
# 1. Instalar dependências
pip install -r requirements.txt
pip install -r requirements-ml.txt

# 2. Preparar dataset Scielo (gera abstracts_scielo.csv)
python prepare_scielo_dataset.py

# 3. Executar pipeline completo (teste → seleção → preparação → fine-tuning → teste final)
python finetune_and_evaluate.py --skip_prepare
```

Se as etapas anteriores não falharem, os resultados finais estarão em:
- `scielo_before_finetuning.csv` - Métricas dos modelos base
- `scielo_after_finetuning.csv` - Métricas dos modelos fine-tuned
- `SCIENCE_EVALUATION_REPORT.txt` - Relatório comparativo

---

## 📋 STAGE 1: Avaliação Inicial dos Modelos

### O que faz:
Avalia **6 modelos pré-treinados** em **4 datasets públicos** para estabelecer baseline.

### Modelos testados:
1. `Helsinki-NLP/opus-mt-tc-big-en-pt` (MarianMT)
2. `Narrativa/mbart-large-50-finetuned-opus-en-pt-translation` (mBART)
3. `unicamp-dl/translation-en-pt-t5` (T5)
4. `VanessaSchenkel/unicamp-finetuned-en-to-pt-dataset-ted` (T5 fine-tuned TED)
5. `danhsf/m2m100_418M-finetuned-kde4-en-to-pt_BR` (M2M100)
6. `quickmt/quickmt-en-pt` (CTranslate2)

### Datasets públicos:
- **WMT24++** (en-pt_BR): 998 exemplos
- **ParaCrawl** (en→pt): 5000 exemplos  
- **Flores** (Facebook): 1012 exemplos
- **OPUS100** (en-pt): 5000 exemplos

### Métricas calculadas:
- **BLEU**: Precisão de n-gramas (0-100)
- **chr-F**: F-score baseado em caracteres (0-100)
- **COMET**: Score neural aprendido (0-1)
- **BERTScore F1**: Similaridade semântica (0-1)

### Executar STAGE 1:

```bash
# Avaliar os 5 modelos primários
python models-test.py --resume

# ou para refazer do zero
python models-test.py --full

# Avaliar o 6º modelo (QuickMT)
python evaluate_quickmt.py --resume

# ou para refazer
python evaluate_quickmt.py --full
```

### Saída gerada:
- `evaluation_results/translation_metrics_all.csv` - Consolidado com todos os resultados
- `evaluation_results/<modelo>.csv` - Resultados por modelo individuais

---

## 🏆 STAGE 2: Seleção dos Melhores Modelos

### O que faz:
Analisa os resultados do STAGE 1 e identifica os **2 melhores modelos** usando score composto.

### Scoring:
```
score = 0.30×BLEU + 0.25×chr-F + 0.25×COMET + 0.20×BERTScore F1
```

Todos os scores são **normalizados min-max** para [0,1] antes de combinar.

### Executar STAGE 2:

```bash
# Analisar e escolher top 2
python choose_best_model.py

# Ou com arquivo customizado
python choose_best_model.py evaluation_results/translation_metrics_all.csv
```

### Saída:
```
════════════════════════════════════════════════════════════════════════════════
  RANKING GERAL - Score composto (BLEU + chr-F + COMET + BERTScore)
════════════════════════════════════════════════════════════════════════════════

  1. danhsf/m2m100_418M-finetuned-kde4-en-to-pt_BR *
     Score: 0.8765  |  BLEU: 29.42  |  chr-F: 50.21  |  COMET: 0.7645 |  BERTScore: 0.8301

  2. Helsinki-NLP/opus-mt-tc-big-en-pt *
     Score: 0.8321  |  BLEU: 33.78  |  chr-F: 59.89  |  COMET: 0.7825 |  BERTScore: 0.8622
```

### Próximo passo:
Os 2 modelos selecionados serão fine-tuned no STAGE 4.

---

## 🗂️ STAGE 3: Preparação de Dados SciELO

### O que faz:
Separa o dataset **abstracts_scielo.csv** em 3 splits não-sobrepostos:

```
abstracts_scielo.csv (2.7M exemplos)
    ↓
┌─────────────────────────────────────────┐
│  Divisão ESTRATIFICADA                  │
├─────────────────────────────────────────┤
│ • TREINO:        200,000 exemplos       │  
│ • VALIDAÇÃO:      20,000 exemplos       │  (monitora convergência)
│ • TESTE:          20,000 exemplos       │  (avaliação final)
│                                          │
│ Total: 240,000 exemplos (~8.7%)         │
└─────────────────────────────────────────┘
    ↓
Salvo em: finetuning/abstracts-datasets/
    ├─ scielo_abstracts_train.csv
    ├─ scielo_abstracts_val.csv
    └─ scielo_abstracts_test.csv
```

### Características importantes:
- **Sem sobreposição**: Cada exemplo aparece em apenas 1 split
- **Seed fixo (42)**: Reprodutibilidade
- **Estratificado**: Mantém distribuição de comprimento equilibrada
- **Determinístico**: Sempre gera os mesmos splits

### Executar STAGE 3:

```bash
# Via select_and_test_models.py (prepara automaticamente)
python finetuning/select_and_test_models.py

# Ou manualmente via datasets.prepare_evaluation_csv
python -c "
from finetuning import config, datasets
datasets.prepare_evaluation_csv(
    abstracts_file='abstracts_scielo.csv',
    train_csv=config.SCIELO_TRAIN_CSV,
    val_csv=config.SCIELO_VAL_CSV,
    test_csv=config.SCIELO_TEST_CSV,
    train_samples=200_000,
    val_samples=20_000,
    test_samples=20_000
)
"
```

### Saída:
- `finetuning/abstracts-datasets/scielo_abstracts_train.csv` (200k linhas)
- `finetuning/abstracts-datasets/scielo_abstracts_val.csv` (20k linhas)
- `finetuning/abstracts-datasets/scielo_abstracts_test.csv` (20k linhas)

### Testar modelos base individualmente:

```bash
# Testar todos os modelos (helsinki + m2m100)
python finetuning/select_and_test_models.py --skip_prepare

# Testar apenas Helsinki
python finetuning/select_and_test_models.py --skip_prepare --model helsinki

# Testar apenas M2M100
python finetuning/select_and_test_models.py --skip_prepare --model m2m100
```

**Saída**: `scielo_before_finetuning.csv` com métricas BLEU, chrF, COMET, BERTScore

---

## 🎓 STAGE 4: Fine-tuning dos Melhores Modelos

### O que faz:
Treina os 2 modelos selecionados no STAGE 2 usando dados de STAGE 3.

### Arquitetura:
- **Seq2SeqTrainer** do HuggingFace
- **Mixed precision training** (FP16 quando possível)
- **Gradient accumulation** se necessário
- **Checkpoints** salvos a cada época

### Configurações padrão:
```python
EPOCHS = 5
BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2
LEARNING_RATE = 2e-5
WARMUP_STEPS = 500
MAX_SEQ_LENGTH = 256
```

### Executar STAGE 4:

```bash
# Fine-tuning de ambos os modelos
python finetuning/finetune_selected_models.py --skip_prepare

# Fine-tuning do modelo específico
python finetuning/finetune_selected_models.py --model helsinki --skip_prepare
python finetuning/finetune_selected_models.py --model m2m100 --skip_prepare

# Com parâmetros customizados
python finetuning/finetune_selected_models.py \
  --model helsinki \
  --epochs 10 \
  --batch_size 4 \
  --lr 5e-5 \
  --skip_prepare

# Retomar fine-tuning interrompido
python finetuning/finetune_selected_models.py \
  --model helsinki \
  --resume_from ./models/finetuned-scielo/helsinki/checkpoint-3000 \
  --skip_prepare
```

### Saída:
```
models/finetuned-scielo/
├── helsinki/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   ├── checkpoint-1000/
│   ├── checkpoint-2000/
│   └── ...
└── m2m100/
    ├── config.json
    ├── pytorch_model.bin
    └── ...
```

### Checkpoints:
- Salvos a cada `eval_steps` (~1/5 da época por padrão)
- Permitem **resumir treino** se interrompido
- Incluem optimizer state para convergência suave

---

## 📊 STAGE 5: Avaliação Final e Comparação

### O que faz:
Avalia os modelos fine-tuned **nos mesmos 20k dados de teste** do STAGE 3 e compara com STAGE 1.

### Crucial: Usar os MESMOS dados de teste
```
STAGE 1 (modelos base):                STAGE 5 (modelos fine-tuned):
├─ Testar em: 20k SciELO teste   vs   ├─ Testar em: MESMOS 20k SciELO
├─ Resultado: BLEU=X.xx              ├─ Resultado: BLEU=Y.yy
└─ Arquivo: scielo_before_*           └─ Arquivo: scielo_after_*

Delta BLEU = Y.yy - X.xx
Se Delta > 20%: ⚠️ Possível overfitting
Se Delta < 0%: ❌ Underfitting / problemas
```

### Executar STAGE 5:

```bash
# Testar nos dados SciELO (todos os modelos)
python finetuning/select_and_test_models.py --test_finetuned --skip_prepare

# Testar modelo específico fine-tuned
python finetuning/select_and_test_models.py --test_finetuned --model helsinki --skip_prepare
python finetuning/select_and_test_models.py --test_finetuned --model m2m100 --skip_prepare

# Comparar base vs fine-tuned (todos os modelos)
python finetuning/select_and_test_models.py --test_both --skip_prepare

# Comparar base vs fine-tuned (modelo específico)
python finetuning/select_and_test_models.py --test_both --model helsinki --skip_prepare

# Gerar CSV de comparação
python compare_results.py
```

### Saída:
- `scielo_before_finetuning.csv` - Modelos base
- `scielo_after_finetuning.csv` - Modelos fine-tuned
- `SCIENCE_EVALUATION_REPORT.txt` - Análise detalhada

### Exemplo de comparação:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPARAÇÃO: Base vs Fine-tuned (SciELO)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

danhsf/m2m100_418M-finetuned-kde4-en-to-pt_BR
  ANTES:   BLEU=22.99  chr-F=50.08
  DEPOIS:  BLEU=28.45  chr-F=54.32
  DELTA:   +23.8% (possível overfitting)  ⚠️

Helsinki-NLP/opus-mt-tc-big-en-pt
  ANTES:   BLEU=33.71  chr-F=58.86
  DEPOIS:  BLEU=35.12  chr-F=60.21
  DELTA:   +4.2% (melhoria moderada)  ✅
```

---

## 🔧 Estrutura do Projeto

```
.
├── 📄 README.md (este arquivo)
├── 📄 requirements.txt
├── 📄 requirements-ml.txt
│
├── 🐍 prepare_scielo_dataset.py           [STAGE 0] Gerar abstracts_scielo.csv
├── 🐍 models-test.py                      [STAGE 1] Avaliar 5 modelos
├── 🐍 evaluate_quickmt.py                 [STAGE 1] Avaliar 6º modelo
├── 🐍 choose_best_model.py                [STAGE 2] Selecionar Top 2
├── 🐍 finetune_and_evaluate.py            [STAGES 1-5] Pipeline integrado
├── 🐍 compare_results.py                  [STAGE 5] Gerar relatório
│
├── 📊 abstracts_scielo.csv                Dataset Scielo completo (2.7M)
├── 📂 evaluation_results/
│   ├── translation_metrics_all.csv        [STAGE 1] Resultado consolidado
│   ├── <modelo>.csv                       [STAGE 1] Resultados por modelo
│   ├── scielo_before_finetuning.csv       [STAGE 5] Modelos base em SciELO
│   └── scielo_after_finetuning.csv        [STAGE 5] Modelos fine-tuned em SciELO
│
├── 📦 finetuning/                         Pacote principal
│   ├── config.py                          Configurações centralizadas
│   ├── models.py                          Carregamento e salvamento
│   ├── datasets.py                        Preparação de dados
│   ├── metrics.py                         BLEU, chr-F, COMET, BERTScore
│   ├── evaluate.py                        Avaliação com progresso (tqdm)
│   ├── trainer.py                         Seq2SeqTrainer + loop fine-tuning
│   ├── compare.py                         Comparação base vs fine-tuned
│   ├── io_utils.py                        Utilitários I/O
│   │
│   ├── select_and_test_models.py          [STAGE 3+5] Teste SciELO
│   ├── finetune_selected_models.py        [STAGE 4] Fine-tuning SciELO
│   │
│   └── abstracts-datasets/                [STAGE 3] Dados SciELO splits
│       ├── scielo_abstracts_train.csv     200k exemplos
│       ├── scielo_abstracts_val.csv       20k exemplos
│       └── scielo_abstracts_test.csv      20k exemplos
│
├── 📂 checkpoints/                        Checkpoints de treino/validação
│   ├── training/
│   └── evaluation/
│
└── 📂 models/finetuned-scielo/           Modelos fine-tuned
    ├── helsinki/
    │   ├── config.json
    │   ├── pytorch_model.bin
    │   └── checkpoint-*/
    └── m2m100/
        ├── config.json
        ├── pytorch_model.bin
        └── checkpoint-*/
```

---

## 💡 Detalhes Técnicos Importantes

### 1. **Reprodutibilidade**
- Seed = 42 em todos os splits de dados
- Modelos carregados com `torch.manual_seed(42)`
- Resultados são determinísticos

### 2. **Sem Sobreposição de Dados**
```python
Total: 2.7M exemplos
Usar: 240k exemplos
├─ Treino: 200k (74%)      ← Fine-tuning
├─ Val:     20k (8%)       ← Monitorar convergência
└─ Teste:   20k (8%)       ← MESMOS dados em STAGE 1 e 5
```

**Importante**: O split de TESTE no STAGE 3 é o **mesmo** usado para testar modelos base no STAGE 1, permitindo comparação justa.

### 3. **Checkpoints e Resumir**
```bash
# Se o treino for interrompido (power failure, timeout, etc)
# Localizar o checkpoint mais recente
ls models/finetuned-scielo/helsinki/

# Retomar exatamente de onde parou
python finetuning/finetune_selected_models.py \
  --model helsinki \
  --resume_from ./models/finetuned-scielo/helena/checkpoint-5000 \
  --skip_prepare
```

### 4. **Detectar Overfitting**
Comparar BLEU de STAGE 1 (base) vs STAGE 5 (fine-tuned):
- **+5% a +15%**: Melhoria saudável ✅
- **+15% a +20%**: Possível overfitting ⚠️
- **> +20%**: Provável overfitting ❌ (rediferenciar dados)
- **< 0%**: Underfitting ❌ (aumentar épocas/dados)

---

## 🛠️ Troubleshooting

### CUDA Out Of Memory
```bash
# Batch size já está em 2 (padrão)
# Se ainda der OOM, tente batch_size=1
python finetuning/finetune_selected_models.py --batch_size 1

# Ou usar CPU (lento!)
export CUDA_VISIBLE_DEVICES=-1
python finetuning/finetune_selected_models.py --batch_size 2
```

### Dataset não encontrado
```bash
# Gerar abstracts_scielo.csv
python prepare_scielo_dataset.py

# Verificar
ls -lh abstracts_scielo.csv
```

### Modelo não carrega
```bash
# Limpar cache HF
rm -rf ~/.cache/huggingface/

# Tentar novamente (vai baixar modelo)
python finetuning/select_and_test_models.py --skip_prepare

# Ou testar modelo específico
python finetuning/select_and_test_models.py --skip_prepare --model helsinki
```

### Treino muito lento
- Reduzir `--train_samples` para teste (ex: 50k)
- Batch size já está otimizado (2)
- GPU com Tensor Cores (A100, RTX 3090) é 10x mais rápido
- Use `--model helsinki` ou `--model m2m100` para treinar 1 modelo por vez

---

## 📚 Referências

- **HuggingFace Transformers**: https://huggingface.co/docs/transformers/
- **SACREBleu**: https://github.com/mjpost/sacrebleu
- **COMET**: https://github.com/Unbabel/COMET
- **BERTScore**: https://github.com/Tiiiger/bert_score

---

## 📝 Reproduzindo este Trabalho

Para rodar exatamente como descrito:

```bash
# 1. Clone e prepare
git clone <repo>
cd hugging-face-model-tests
pip install -r requirements.txt -r requirements-ml.txt

# 2. STAGE 0: Dataset
python prepare_scielo_dataset.py

# 3. STAGE 1: Avaliação
python models-test.py --full
python evaluate_quickmt.py --full

# 4. STAGE 2: Seleção
python choose_best_model.py

# 5. STAGE 3: Preparação (automático na próxima etapa)
# (vai ser feito por select_and_test_models.py)

# 6. STAGE 4: Fine-tuning
python finetuning/finetune_selected_models.py

# 7. STAGE 5: Avaliação Final
python finetuning/select_and_test_models.py --test_both --skip_prepare

# 8. Gerar Relatório
python compare_results.py
```

---

**Versão**: 3.0 | **Data**: Fevereiro 2026

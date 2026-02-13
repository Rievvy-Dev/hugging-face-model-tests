# Avaliacao de Modelos de Traducao EN-PT (TCC)

Repositorio para avaliacao sistematica de modelos de **traducao automatica ingles -> portugues**, usando metricas automaticas e multiplos datasets.

## Metricas

| Metrica | Tipo | Descricao |
|---------|------|-----------|
| **BLEU** | N-gram | Precisao de n-gramas entre traducao e referencia |
| **chr-F** | N-gram (caractere) | F-score baseado em n-gramas de caracteres |
| **COMET** | Neural | Score aprendido (modelo `Unbabel/wmt22-comet-da`). Requer `unbabel-comet` |
| **BERTScore F1** | Neural | Similaridade semantica via embeddings BERT. Requer `bert-score` |

## Modelos avaliados

| Modelo | Framework |
|--------|-----------|
| Helsinki-NLP/opus-mt-tc-big-en-pt | MarianMT (Transformers) |
| Narrativa/mbart-large-50-finetuned-opus-en-pt-translation | mBART (Transformers) |
| unicamp-dl/translation-en-pt-t5 | T5 (Transformers) |
| VanessaSchenkel/unicamp-finetuned-en-to-pt-dataset-ted | T5 (Transformers) |
| danhsf/m2m100_418M-finetuned-kde4-en-to-pt_BR | M2M100 (Transformers) |
| aimped/nlp-health-translation-base-en-pt | MarianMT (Transformers) |
| quickmt/quickmt-en-pt | CTranslate2 (`evaluate_quickmt.py`) |

## Datasets

| Dataset | Split | Amostras (max) |
|---------|-------|----------------|
| WMT24++ (`google/wmt24pp`, en-pt_BR) | train | 5000 |
| ParaCrawl (`para_crawl`, enpt) | train | 5000 |
| Flores (`facebook/flores`, eng_Latn-por_Latn) | devtest | 1012 |
| OPUS100 (`opus100`, en-pt) | test | 5000 |

---

## Instalacao

### 1. Dependencias basicas

```bash
pip install transformers datasets evaluate tqdm psutil sacrebleu
```

### 2. Metricas neurais (COMET + BERTScore)

```bash
pip install unbabel-comet bert-score
```

> Se encontrar erro `ModuleNotFoundError: No module named 'pkg_resources'`, rode:
> ```bash
> pip install "setuptools<81"
> ```

### 3. GPU (recomendado)

O `pip install torch` padrao instala apenas a versao CPU. Para GPU NVIDIA:

```bash
# Verifique se a GPU esta visivel:
python check_gpu.py

# Se CUDA=False, instale PyTorch com CUDA (escolha a versao do seu driver):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Outras opcoes: `cu121` (CUDA 12.1) ou `cu118` (CUDA 11.8). Veja [pytorch.org](https://pytorch.org/get-started/locally/).

### 4. QuickMT (opcional)

```bash
git clone https://github.com/quickmt/quickmt.git
pip install ./quickmt/
pip install huggingface_hub
```

---

## Scripts

| Script | Descricao |
|--------|-----------|
| `models-test.py` | Launcher da avaliacao principal. Delega para `evaluation/run.py`. |
| `evaluate_quickmt.py` | Avaliacao do QuickMT (CTranslate2) nos mesmos datasets. |
| `choose_best_model.py` | Analisa resultados e sugere os **2 melhores modelos** para fine-tuning. |
| `fill_missing_metrics.py` | Preenche COMET/BERTScore em CSVs que ja tem traducoes (sem retraduzir). |
| `helsinki-trainning.py` | Fine-tuning do Helsinki (MarianMT) em OPUS100 + Tatoeba. |
| `test_helsinki_finetuned.py` | Avaliacao do modelo fine-tunado em varios datasets. |
| `check_gpu.py` | Verifica se PyTorch enxerga a GPU. |

---

## Como usar

### Passo 1 - Avaliacao principal (todos os modelos Transformers)

```bash
# Modo resume (padrao): pula combinacoes ja avaliadas
python models-test.py

# Modo full: apaga CSVs anteriores e roda TUDO do zero
python models-test.py --full
```

- Resultados salvos em `evaluation_results/translation_metrics_all.csv` (agregado) e `evaluation_results/<modelo>.csv` (por modelo).
- O modo resume permite **cancelar e continuar** de onde parou (Ctrl+C e rodar de novo).
- O `--full` remove todos os CSVs da pasta `evaluation_results/` antes de comecar.

### Passo 2 - Avaliacao do QuickMT (opcional)

```bash
# Modo resume (padrao)
python evaluate_quickmt.py

# Modo full: apaga CSV anterior e roda do zero
python evaluate_quickmt.py --full
```

Resultados em `evaluation_results/resultados_quickmt.csv`.

### Passo 3 - Escolher os melhores modelos

```bash
python choose_best_model.py
```

Mostra:
- **Ranking geral** com score composto (BLEU 30% + chr-F 25% + COMET 25% + BERTScore 20%)
- **Rankings individuais** por cada metrica
- **Top 2 modelos** recomendados para fine-tuning com sugestoes de comando

### Passo 4 - Preencher metricas faltantes (opcional)

Se voce tem CSVs antigos sem COMET/BERTScore:

```bash
python fill_missing_metrics.py evaluation_results/translation_metrics_all.csv
python fill_missing_metrics.py resultados_traducao_wmt_paracrawl.csv --output evaluation_results/translation_metrics_preenchido.csv
```

### Passo 5 - Fine-tuning (opcional)

```bash
# Helsinki (MarianMT) em OPUS100 + Tatoeba
python helsinki-trainning.py --epochs 3 --output_dir ./models/opus-mt-en-pt-finetuned

# Com corpus cientifico extra
python helsinki-trainning.py --extra_csv ./corpus_cientifico.csv --epochs 3
```

---

## Arquivos de resultados

| Arquivo | Descricao |
|---------|-----------|
| `evaluation_results/translation_metrics_all.csv` | Saida principal: todos os modelos x datasets (BLEU, chr-F, COMET, BERTScore). |
| `evaluation_results/resultados_quickmt.csv` | Resultados do QuickMT (CTranslate2). |
| `evaluation_results/<modelo>.csv` | CSV individual por modelo. |
| `evaluation_results/translation_metrics_preenchido.csv` | Metricas preenchidas via `fill_missing_metrics`. |
| `resultados_finetuned.csv` | Resultados do modelo fine-tunado (Helsinki) - dados historicos. |
| `resultados_traducao_final.csv` | Traducoes de varios modelos - dados historicos (BLEU, chr-F). |
| `resultados_traducao_wmt_paracrawl.csv` | Traducoes WMT + Paracrawl - dados historicos. |
| `resultados_traducao_flores_wmt.csv` | Flores + WMT - dados historicos. |

> **Nota:** os arquivos `resultados_*.csv` na raiz contem dados historicos de avaliacoes anteriores. Nao remova.

---

## Gerenciamento de memoria GPU

O pipeline foi otimizado para GPUs com VRAM limitada (ex: RTX 2050, 4GB):

- O modelo de traducao e movido para CPU antes de calcular COMET e BERTScore.
- COMET e BERTScore sao calculados sequencialmente, liberando VRAM entre eles.
- O modelo de traducao volta para GPU automaticamente para o proximo dataset.

Se mesmo assim ocorrer OOM, reduza `batch_size` em `evaluation/config.py`.

---

## Estrutura do projeto

```
hugging-face-model-tests/
  models-test.py              # Launcher principal
  evaluate_quickmt.py         # QuickMT (CTranslate2)
  choose_best_model.py        # Ranking e top 2 modelos
  fill_missing_metrics.py     # Preenche COMET/BERTScore em CSVs existentes
  helsinki-trainning.py        # Fine-tuning Helsinki
  test_helsinki_finetuned.py   # Avaliacao do fine-tunado
  check_gpu.py                # Verificacao de GPU
  evaluation/                 # Pacote de avaliacao
    __init__.py
    config.py                 # Configuracoes (modelos, datasets, limites)
    run.py                    # Loop principal de avaliacao
    metrics.py                # BLEU, chr-F, COMET, BERTScore
    datasets.py               # Carregamento de datasets
    models_loader.py          # Carregamento de modelos
    io_utils.py               # Funcoes de I/O (CSV)
    fill_missing_metrics.py   # Versao modular do fill_missing_metrics
  evaluation_results/         # CSVs de saida
  resultados_*.csv            # CSVs historicos
```

---

## Fine-tuning em corpus cientifico (TCC)

Para traducao de abstracts e artigos cientificos, o modelo pode ser fine-tunado em corpora cientificos:

- **SciELO** (en-pt): artigos e abstracts alinhados. Ref: *Soares et al., LREC 2018*. Disponivel em [figshare](https://figshare.com/s/091fcaf8ad66a3304e90).
- Integrar como CSV (colunas `source`/`target` ou `en`/`pt`) no pipeline de treino via `--extra_csv`.

---

## Licenca

Projeto aberto (MIT).

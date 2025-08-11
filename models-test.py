# -*- coding: utf-8 -*-
import torch
import psutil
import gc
import csv
import os
import time
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import evaluate

# ==================== Configs ====================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando GPU? {torch.cuda.is_available()}")
print(f"Dispositivo: {device}")

# Tamanho do "teste" do ParaCrawl
PARACRAWL_N = 5000
# True = usa fatia train[:N] (mais rápido, determinístico, não precisa baixar tudo)
# False = baixa o train completo, embaralha e seleciona N (mais diverso, mas pesado)
PARACRAWL_FAST_SLICE = True

# ==================== Métricas ====================
bleu_metric = evaluate.load("bleu")
chrf_metric = evaluate.load("chrf")

# ==================== Modelos ====================
MODEL_NAMES = [
    "Helsinki-NLP/opus-mt-tc-big-en-pt",
    "Narrativa/mbart-large-50-finetuned-opus-en-pt-translation",
    "unicamp-dl/translation-en-pt-t5",
    "VanessaSchenkel/unicamp-finetuned-en-to-pt-dataset-ted",
]

# ==================== Loaders ====================
def load_wmt24pp():
    return (load_dataset("google/wmt24pp", "en-pt_BR", split="train")
            .shuffle(seed=42).select(range(998)))

def wmt24pp_total():
    return load_dataset("google/wmt24pp", "en-pt_BR", split="train").num_rows

def load_paracrawl():
    if PARACRAWL_FAST_SLICE:
        # fatia determinística do train (rápido)
        return load_dataset(
            "para_crawl", "enpt", split=f"train[:{PARACRAWL_N}]",
            trust_remote_code=True
        )
    else:
        # baixa tudo, embaralha e pega N (mais diverso; pesado)
        ds = load_dataset("para_crawl", "enpt", split="train", trust_remote_code=True)
        return ds.shuffle(seed=42).select(range(PARACRAWL_N))

def paracrawl_total():
    # evitar custo de contar o train inteiro
    return ""

DATASETS_INFO = [
    ("wmt24pp",  load_wmt24pp,   wmt24pp_total),
    ("paracrawl", load_paracrawl, paracrawl_total),
]

# ==================== Batch sizes ====================
BATCH_SIZE_BY_MODEL_DATASET = {
    ("wmt24pp", "Helsinki-NLP/opus-mt-tc-big-en-pt"): 1,
    ("wmt24pp", "Narrativa/mbart-large-50-finetuned-opus-en-pt-translation"): 1,
    ("wmt24pp", "unicamp-dl/translation-en-pt-t5"): 1,
    ("wmt24pp", "VanessaSchenkel/unicamp-finetuned-en-to-pt-dataset-ted"): 1,

    ("paracrawl", "Helsinki-NLP/opus-mt-tc-big-en-pt"): 1,
    ("paracrawl", "Narrativa/mbart-large-50-finetuned-opus-en-pt-translation"): 1,
    ("paracrawl", "unicamp-dl/translation-en-pt-t5"): 1,
    ("paracrawl", "VanessaSchenkel/unicamp-finetuned-en-to-pt-dataset-ted"): 1,
}

# ==================== Saída ====================
OUTPUT_FILE = "resultados_traducao_wmt_paracrawl.csv"

def save_header_if_needed(filename, header):
    if not os.path.exists(filename):
        with open(filename, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

# ==================== Helpers ====================
def _extract_texts(batch, dataset_name):
    """Retorna (srcs, tgts) de acordo com o dataset."""
    if dataset_name == "wmt24pp":
        srcs = [ex.get("source", "") for ex in batch]
        tgts = [ex.get("target", "") for ex in batch]
    elif dataset_name == "paracrawl":
        if "translation" in batch[0]:
            srcs = [ex["translation"].get("en", "") for ex in batch]
            tgts = [ex["translation"].get("pt", "") for ex in batch]
        else:
            srcs = [ex.get("source", "") for ex in batch]
            tgts = [ex.get("target", "") for ex in batch]
    else:
        srcs, tgts = [], []
    return srcs, tgts

def get_dataset_stats(dataset, dataset_name):
    """Estatísticas sobre os TARGETS."""
    if dataset_name == "wmt24pp":
        textos = [ex.get("target", "") for ex in dataset]
    elif dataset_name == "paracrawl":
        if "translation" in dataset[0]:
            textos = [ex["translation"].get("pt", "") for ex in dataset]
        else:
            textos = [ex.get("target", "") for ex in dataset]
    else:
        textos = []
    num_sent = len(textos)
    total_palavras = sum(len(s.split()) for s in textos)
    media = total_palavras / num_sent if num_sent else 0.0
    return num_sent, total_palavras, media

def get_first_examples(dataset, dataset_name, n=2):
    exemplos = []
    for i in range(min(len(dataset), n)):
        s = dataset[i]
        if dataset_name == "wmt24pp":
            exemplos.append(f"EN: {s.get('source','')}\nPT: {s.get('target','')}")
        elif dataset_name == "paracrawl":
            if "translation" in s:
                exemplos.append(f"EN: {s['translation'].get('en','')}\nPT: {s['translation'].get('pt','')}")
            else:
                exemplos.append(f"EN: {s.get('source','')}\nPT: {s.get('target','')}")
    return " || ".join(exemplos)

# ==================== Avaliação ====================
def compute_metrics(dataset, model, tokenizer, batch_size, dataset_name, model_name):
    model.eval()
    references, hypotheses = [], []
    num_examples = len(dataset)
    erro_msg = ""
    t_start = time.time()
    pbar = tqdm(total=num_examples, desc=f"Processando ({dataset_name}, batch={batch_size})", ncols=100)

    # Suporte opcional para NLLB (se algum dia usar)
    is_nllb = "nllb" in model_name.lower()
    forced_bos_token_id = None
    if is_nllb:
        tokenizer.src_lang = "eng_Latn"
        tokenizer.tgt_lang = "por_Latn"
        forced_bos_token_id = tokenizer.convert_tokens_to_ids("por_Latn")

    for start_idx in range(0, num_examples, batch_size):
        batch = dataset[start_idx:start_idx+batch_size]
        if isinstance(batch, dict):
            batch = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]
        if not batch:
            continue

        srcs, tgts = _extract_texts(batch, dataset_name)
        if not srcs:
            erro_msg += f" | Estrutura inesperada em idx {start_idx}"
            pbar.update(batch_size)
            continue

        try:
            inputs = tokenizer(srcs, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                if is_nllb:
                    out = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id)
                else:
                    out = model.generate(**inputs)
            batch_hyps = [tokenizer.decode(t, skip_special_tokens=True) for t in out]
            hypotheses.extend(batch_hyps)
            references.extend([[t] for t in tgts])
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"❗ OOM em idx {start_idx}, pulando batch...")
                erro_msg += f" | OOM idx {start_idx}"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                raise
        pbar.update(batch_size)
    pbar.close()
    elapsed = time.time() - t_start

    if not references or not hypotheses:
        raise RuntimeError("Nenhuma tradução produzida!")

    bleu_score = bleu_metric.compute(predictions=hypotheses, references=references)["bleu"] * 100
    chrf_score = chrf_metric.compute(predictions=hypotheses, references=references)["score"]
    return bleu_score, chrf_score, elapsed, erro_msg

def print_table(data):
    print("\n" + "-" * 150)
    for metric, value in zip(data["Metric"], data["Valor"]):
        print(f"| {metric.ljust(40)} | {str(value).ljust(100)} |")
    print("-" * 150)

def evaluate_model_on_dataset(dataset_name, dataset, total_real, model_name, model, tokenizer, batch_size=None):
    print(f"\n🗂️  Avaliando dataset: {dataset_name} - modelo: {model_name}")
    exemplos_do_dataset = get_first_examples(dataset, dataset_name, 2)
    num_sentences, num_words, media_palavras = get_dataset_stats(dataset, dataset_name)
    print(f"  - Tamanho real do dataset: {total_real}")
    print(f"  - Total de sentenças (amostradas): {num_sentences}")
    print(f"  - Total de palavras: {num_words}")
    print(f"  - Média de palavras por sentença: {media_palavras:.2f}")

    batch_size = batch_size or 1
    print(f"  - Usando batch size: {batch_size}")

    process = psutil.Process()
    memory_before = process.memory_info().rss
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    try:
        print(f"✅ Modelo carregado em: {device}")
        bleu_score, chrf_score, elapsed_time, erro_msg = compute_metrics(
            dataset, model, tokenizer, batch_size, dataset_name, model_name
        )

        memory_after = process.memory_info().rss
        gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
        gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 ** 2) if torch.cuda.is_available() else 0

        tempo_por_sentenca = elapsed_time / num_sentences if num_sentences else 0.0
        palavras_por_segundo = (num_words / elapsed_time) if elapsed_time > 0 else 0.0

        data = {
            "Metric": [
                "Dataset", "Tamanho Real", "Modelo", "Device", "Batch Size", "Tamanho Dataset (usado)",
                "Exemplos do Dataset", "Total de Sentenças", "Total de Palavras", "Média Palavras por Sentença",
                "Tempo Total", "Tempo por Sentença (s)", "Palavras por Segundo",
                "Uso de Memória (RAM)", "Memória GPU Alocada (MB)", "Memória GPU Reservada (MB)",
                "BLEU", "chr-F", "Erro"
            ],
            "Valor": [
                dataset_name, total_real, model_name, device, batch_size, len(dataset),
                exemplos_do_dataset, num_sentences, num_words, f"{media_palavras:.2f}",
                f"{elapsed_time:.2f}s", f"{tempo_por_sentenca:.4f}", f"{palavras_por_segundo:.2f}",
                f"{(memory_after - memory_before) / (1024 * 1024):.2f} MB",
                f"{gpu_memory_allocated:.2f}", f"{gpu_memory_reserved:.2f}",
                f"{bleu_score:.2f}", f"{chrf_score:.5f}", erro_msg
            ]
        }
    except Exception as e:
        erro_msg = f"{type(e).__name__}: {e}"
        data = {
            "Metric": [
                "Dataset", "Tamanho Real", "Modelo", "Device", "Batch Size", "Tamanho Dataset (usado)",
                "Exemplos do Dataset", "Total de Sentenças", "Total de Palavras", "Média Palavras por Sentença",
                "Tempo Total", "Tempo por Sentença (s)", "Palavras por Segundo",
                "Uso de Memória (RAM)", "Memória GPU Alocada (MB)", "Memória GPU Reservada (MB)",
                "BLEU", "chr-F", "Erro"
            ],
            "Valor": [
                dataset_name, "", model_name, device, batch_size, "",
                exemplos_do_dataset, "", "", "", "", "", "",
                "", "", "", "", "", erro_msg
            ]
        }
        print(f"❌ Erro ao avaliar modelo {model_name} no dataset {dataset_name}: {erro_msg}")

    print_table(data)
    with open(OUTPUT_FILE, mode="a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(data["Valor"])

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ==================== Main ====================
save_header_if_needed(OUTPUT_FILE, [
    "Dataset", "Tamanho Real", "Modelo", "Device", "Batch Size", "Tamanho Dataset (usado)",
    "Exemplos do Dataset", "Total de Sentenças", "Total de Palavras", "Média Palavras por Sentença",
    "Tempo Total", "Tempo por Sentença (s)", "Palavras por Segundo",
    "Uso de Memória (RAM)", "Memória GPU Alocada (MB)", "Memória GPU Reservada (MB)",
    "BLEU", "chr-F", "Erro"
])

# Carregar datasets
DATASETS = {}
DATASET_REAL_SIZE = {}
for name, loader, size_fn in DATASETS_INFO:
    try:
        DATASETS[name] = loader()
        DATASET_REAL_SIZE[name] = size_fn()
        print(f"✅ {name}: {len(DATASETS[name])} amostras (de {DATASET_REAL_SIZE[name]})")
    except Exception as e:
        print(f"❌ Erro ao carregar o dataset '{name}': {e}")

# Carregar modelos/tokenizers
MODELS = {}
TOKENIZERS = {}
for model_name in MODEL_NAMES:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        MODELS[model_name] = model
        TOKENIZERS[model_name] = tokenizer
        print(f"✅ Modelo carregado: {model_name}")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo '{model_name}': {e}")

# Avaliação cruzada
for dataset_name in DATASETS:
    for model_name in MODELS:
        batch_size = BATCH_SIZE_BY_MODEL_DATASET.get((dataset_name, model_name), 1)
        evaluate_model_on_dataset(
            dataset_name,
            DATASETS[dataset_name],
            DATASET_REAL_SIZE.get(dataset_name, ""),
            model_name,
            MODELS[model_name],
            TOKENIZERS[model_name],
            batch_size
        )

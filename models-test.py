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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando GPU? {torch.cuda.is_available()}")
print(f"Dispositivo: {device}")

bleu_metric = evaluate.load("bleu")
chrf_metric = evaluate.load("chrf")

MODEL_NAMES = [
    "Helsinki-NLP/opus-mt-tc-big-en-pt",
    "Narrativa/mbart-large-50-finetuned-opus-en-pt-translation",
    "unicamp-dl/translation-en-pt-t5",
    "VanessaSchenkel/unicamp-finetuned-en-to-pt-dataset-ted"
]

DATASETS_INFO = [
    ("ted_talks",
        lambda: load_dataset("opus100", "en-pt", split="train").shuffle(seed=42).select(range(5000)),
        lambda: load_dataset("opus100", "en-pt", split="train").num_rows),
    ("tatoeba",
        lambda: load_dataset("tatoeba", lang1="en", lang2="pt", trust_remote_code=True)["train"].shuffle(seed=42).select(range(5000)),
        lambda: load_dataset("tatoeba", lang1="en", lang2="pt", trust_remote_code=True)["train"].num_rows),
    ("flores101",
        lambda: load_dataset("facebook/flores", "eng_Latn-por_Latn", split="dev").shuffle(seed=42).select(range(1012)),
        lambda: load_dataset("facebook/flores", "eng_Latn-por_Latn", split="dev").num_rows),
]

BATCH_SIZE_BY_MODEL_DATASET = {
    ("ted_talks", "Helsinki-NLP/opus-mt-tc-big-en-pt"): 1,
    ("ted_talks", "Narrativa/mbart-large-50-finetuned-opus-en-pt-translation"): 1,
    ("ted_talks", "unicamp-dl/translation-en-pt-t5"): 1,
    ("ted_talks", "VanessaSchenkel/unicamp-finetuned-en-to-pt-dataset-ted"): 1,
    ("tatoeba", "Helsinki-NLP/opus-mt-tc-big-en-pt"): 8,
    ("tatoeba", "Narrativa/mbart-large-50-finetuned-opus-en-pt-translation"): 8,
    ("tatoeba", "unicamp-dl/translation-en-pt-t5"): 8,
    ("tatoeba", "VanessaSchenkel/unicamp-finetuned-en-to-pt-dataset-ted"): 8,
    ("flores101", "Helsinki-NLP/opus-mt-tc-big-en-pt"): 4,
    ("flores101", "Narrativa/mbart-large-50-finetuned-opus-en-pt-translation"): 4,
    ("flores101", "unicamp-dl/translation-en-pt-t5"): 4,
    ("flores101", "VanessaSchenkel/unicamp-finetuned-en-to-pt-dataset-ted"): 4,
}

OUTPUT_FILE = "resultados_traducao_final.csv"

def save_header_if_needed(filename, header):
    if not os.path.exists(filename):
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

def get_dataset_stats(dataset):
    if "translation" in dataset[0]:
        textos = [ex["translation"].get("pt", ex["translation"].get("target", "")) for ex in dataset]
    elif "target" in dataset[0]:
        textos = [ex["target"] for ex in dataset]
    else:
        textos = []
    num_sent = len(textos)
    total_palavras = sum(len(s.split()) for s in textos)
    media_palavras = total_palavras / num_sent if num_sent > 0 else 0
    return num_sent, total_palavras, media_palavras

def get_first_examples(dataset, n=2):
    exemplos = []
    for i in range(min(len(dataset), n)):
        sample = dataset[i]
        if "translation" in sample:
            exemplos.append(f"EN: {sample['translation']['en']}\nPT: {sample['translation']['pt']}")
        elif "source" in sample and "target" in sample:
            exemplos.append(f"EN: {sample['source']}\nPT: {sample['target']}")
    return " || ".join(exemplos)

def compute_metrics(dataset, model, tokenizer, batch_size, dataset_name):
    model.eval()
    references, hypotheses = [], []
    num_examples = len(dataset)
    erro_msg = ""
    t_start = time.time()
    pbar = tqdm(total=num_examples, desc=f"Processando ({dataset_name}, batch={batch_size})", ncols=100)
    for start_idx in range(0, num_examples, batch_size):
        batch = dataset[start_idx:start_idx+batch_size]
        if isinstance(batch, dict):
            batch = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]
        elif not isinstance(batch, list):
            batch = list(batch)
        if len(batch) == 0:
            continue
        if "translation" in batch[0]:
            srcs = [ex["translation"]["en"] for ex in batch]
            tgts = [ex["translation"]["pt"] for ex in batch]
        elif "source" in batch[0] and "target" in batch[0]:
            srcs = [ex["source"] for ex in batch]
            tgts = [ex["target"] for ex in batch]
        else:
            erro_msg += f" | Estrutura inesperada em idx {start_idx}"
            pbar.update(batch_size)
            continue
        try:
            inputs = tokenizer(srcs, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                translated = model.generate(**inputs)
            batch_hyps = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
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
    t_end = time.time()
    if not references or not hypotheses:
        raise RuntimeError("Nenhuma tradução produzida! Verifique os campos dos exemplos ou o processamento do dataset.")
    bleu_score = bleu_metric.compute(predictions=hypotheses, references=references)["bleu"] * 100
    chrf_score = chrf_metric.compute(predictions=hypotheses, references=references)["score"]
    elapsed = t_end - t_start
    return bleu_score, chrf_score, elapsed, erro_msg

def print_table(data):
    print("\n" + "-" * 150)
    for metric, value in zip(data["Metric"], data["Valor"]):
        print(f"| {metric.ljust(40)} | {str(value).ljust(100)} |")
    print("-" * 150)

def evaluate_model_on_dataset(dataset_name, dataset, total_real, model_name, model, tokenizer, batch_size=None):
    print(f"\n🗂️  Avaliando dataset: {dataset_name} - modelo: {model_name}")

    # Coletar exemplos para salvar no CSV
    exemplos_do_dataset = get_first_examples(dataset, 2)

    num_sentences, num_words, media_palavras = get_dataset_stats(dataset)
    print(f"  - Tamanho real do dataset: {total_real}")
    print(f"  - Total de sentenças (amostradas): {num_sentences}")
    print(f"  - Total de palavras: {num_words}")
    print(f"  - Média de palavras por sentença: {media_palavras:.2f}")

    if batch_size is None:
        batch_size = 1  # fallback

    print(f"  - Usando batch size: {batch_size}")

    process = psutil.Process()
    memory_before = process.memory_info().rss
    erro_msg = ""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    try:
        print(f"✅ Modelo carregado em: {device}")

        bleu_score, chrf_score, elapsed_time, erro_msg = compute_metrics(
            dataset, model, tokenizer, batch_size, dataset_name)

        memory_after = process.memory_info().rss
        gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
        gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 ** 2) if torch.cuda.is_available() else 0

        tempo_por_sentenca = elapsed_time / num_sentences
        palavras_por_segundo = num_words / elapsed_time

        data = {
            "Metric": [
                "Dataset", "Tamanho Real", "Modelo", "Device", "Batch Size", "Tamanho Dataset (usado)",
                "Exemplos do Dataset",
                "Total de Sentenças", "Total de Palavras", "Média Palavras por Sentença",
                "Tempo Total", "Tempo por Sentença (s)", "Palavras por Segundo",
                "Uso de Memória (RAM)", "Memória GPU Alocada (MB)", "Memória GPU Reservada (MB)",
                "BLEU", "chr-F", "Erro"
            ],
            "Valor": [
                dataset_name, total_real, model_name, device, batch_size, len(dataset),
                exemplos_do_dataset,
                num_sentences, num_words, f"{media_palavras:.2f}",
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
                "Exemplos do Dataset",
                "Total de Sentenças", "Total de Palavras", "Média Palavras por Sentença",
                "Tempo Total", "Tempo por Sentença (s)", "Palavras por Segundo",
                "Uso de Memória (RAM)", "Memória GPU Alocada (MB)", "Memória GPU Reservada (MB)",
                "BLEU", "chr-F", "Erro"
            ],
            "Valor": [
                dataset_name, "", model_name, device, batch_size, "",
                exemplos_do_dataset,
                "", "", "",
                "", "", "",
                "", "", "",
                "", "", erro_msg
            ]
        }
        print(f"❌ Erro ao avaliar modelo {model_name} no dataset {dataset_name}: {erro_msg}")

    print_table(data)
    with open(OUTPUT_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(data["Valor"])

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ---------- Execução principal ------------

save_header_if_needed(OUTPUT_FILE, [
    "Dataset",
    "Tamanho Real",
    "Modelo",
    "Device",
    "Batch Size",
    "Tamanho Dataset (usado)",
    "Exemplos do Dataset",
    "Total de Sentenças",
    "Total de Palavras",
    "Média Palavras por Sentença",
    "Tempo Total",
    "Tempo por Sentença (s)",
    "Palavras por Segundo",
    "Uso de Memória (RAM)",
    "Memória GPU Alocada (MB)",
    "Memória GPU Reservada (MB)",
    "BLEU",
    "chr-F",
    "Erro"
])

# Carregar todos os datasets só uma vez
DATASETS = {}
DATASET_REAL_SIZE = {}
for dataset_name, dataset_loader, dataset_total_size_fn in DATASETS_INFO:
    try:
        DATASETS[dataset_name] = dataset_loader()
        DATASET_REAL_SIZE[dataset_name] = dataset_total_size_fn()
    except Exception as e:
        print(f"❌ Erro ao carregar o dataset '{dataset_name}': {e}")

# Carregar todos os modelos/tokenizers só uma vez
MODELS = {}
TOKENIZERS = {}
for model_name in MODEL_NAMES:
    try:
        TOKENIZERS[model_name] = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        MODELS[model_name] = model
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

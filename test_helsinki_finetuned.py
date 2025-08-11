# -*- coding: utf-8 -*-
import torch
import psutil
import gc
import csv
import os
import time
from datasets import load_dataset
from datasets.utils.logging import set_verbosity_error
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
import evaluate

set_verbosity_error()

# ====== CUDA obrigatório ======
def require_cuda():
    if not torch.cuda.is_available():
        raise SystemExit("❌ CUDA não disponível! Este script exige GPU com CUDA.")
    device = torch.device("cuda")
    print(f"✅ Usando dispositivo: {device} ({torch.cuda.get_device_name(0)})")
    return device

device = require_cuda()

# ====== CONFIG ======
MODEL_DIR = "./models/opus-mt-en-pt-finetuned"  # ajuste se necessário
OUTPUT_FILE = "resultados_finetuned.csv"
USE_FLORES = True
MAX_LEN = 128
SEED = 42

# --- Novos controles para o ParaCrawl ---
PARACRAWL_N = 5000
PARACRAWL_FAST_SLICE = True  # True: train[:N] (rápido e determinístico) | False: baixa tudo e embaralha

bleu_metric = evaluate.load("bleu")
chrf_metric = evaluate.load("chrf")

# ====== DATASETS ======
def load_wmt24pp():
    # wmt24pp (en-pt_BR) — usamos uma amostra determinística para manter tempo razoável
    # se quiser tudo, troque .select(range(998)) por sem select (cuidado com tempo/memória)
    return (load_dataset("google/wmt24pp", "en-pt_BR", split="train")
            .shuffle(seed=SEED).select(range(998)))

def wmt24pp_total():
    return load_dataset("google/wmt24pp", "en-pt_BR", split="train").num_rows

def load_paracrawl():
    if PARACRAWL_FAST_SLICE:
        return load_dataset(
            "para_crawl", "enpt", split=f"train[:{PARACRAWL_N}]",
            trust_remote_code=True
        )
    else:
        ds = load_dataset("para_crawl", "enpt", split="train", trust_remote_code=True)
        return ds.shuffle(seed=SEED).select(range(PARACRAWL_N))

def paracrawl_total():
    # manter vazio para não forçar contagem do train completo
    return ""

DATASETS_INFO = [
    # ---- já existentes ----
    ("ted_talks",
        lambda: load_dataset("opus100", "en-pt", split="train").shuffle(seed=SEED).select(range(5000)),
        lambda: load_dataset("opus100", "en-pt", split="train").num_rows),
    ("tatoeba",
        lambda: load_dataset("tatoeba", lang1="en", lang2="pt", trust_remote_code=True)["train"].shuffle(seed=SEED).select(range(5000)),
        lambda: load_dataset("tatoeba", lang1="en", lang2="pt", trust_remote_code=True)["train"].num_rows),

    # ---- novos: WMT24++ e ParaCrawl ----
    ("wmt24pp", load_wmt24pp, wmt24pp_total),
    ("paracrawl", load_paracrawl, paracrawl_total),
]

if USE_FLORES:
    DATASETS_INFO.append(
        ("flores101",
         lambda: load_dataset("facebook/flores", "eng_Latn-por_Latn", split="dev").shuffle(seed=SEED).select(range(1012)),
         lambda: load_dataset("facebook/flores", "eng_Latn-por_Latn", split="dev").num_rows)
    )

# batch sizes (pode ajustar conforme VRAM)
BATCH_SIZE_BY_DATASET = {
    "ted_talks": 1,
    "tatoeba": 8,
    "flores101": 4,
    "wmt24pp": 1,   # frases mais longas; mantenha conservador
    "paracrawl": 1, # diversidade alta; começa pequeno para evitar OOM
}

def save_header_if_needed(filename, header):
    if not os.path.exists(filename):
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

def prepare_batch(batch):
    def _get_pair(ex):
        # 1) preferencial: translation dict
        if "translation" in ex and isinstance(ex["translation"], dict):
            tr = ex["translation"]
            en = tr.get("en") or tr.get("source") or tr.get("sentence_en") or tr.get("text_en")
            pt = tr.get("pt") or tr.get("target") or tr.get("sentence_pt") or tr.get("text_pt")
            if en is not None and pt is not None:
                return en, pt
        # 2) outros formatos comuns
        if "source" in ex and "target" in ex:
            return ex["source"], ex["target"]
        if "en" in ex and "pt" in ex:
            return ex["en"], ex["pt"]
        if "sentence1" in ex and "sentence2" in ex:
            return ex["sentence1"], ex["sentence2"]
        # 3) heurística por prefixo
        en_key = next((k for k in ex.keys() if k.lower().startswith("en")), None)
        pt_key = next((k for k in ex.keys() if k.lower().startswith("pt")), None)
        if en_key and pt_key:
            return ex[en_key], ex[pt_key]
        return None, None

    srcs, tgts = [], []
    for ex in batch:
        en, pt = _get_pair(ex)
        if en is not None and pt is not None:
            srcs.append(en)
            tgts.append(pt)
    return srcs, tgts

def compute_metrics(dataset, model, tokenizer, batch_size, dataset_name):
    print(f"   ▶️ Avaliando {dataset_name} com batch_size={batch_size}…")
    model.eval()
    references, hypotheses = [], []
    num_examples = len(dataset)
    t_start = time.time()
    pbar = tqdm(total=num_examples, desc=f"Processando ({dataset_name})", ncols=100)
    for start_idx in range(0, num_examples, batch_size):
        batch = dataset[start_idx:start_idx+batch_size]
        if isinstance(batch, dict):
            batch = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]
        elif not isinstance(batch, list):
            batch = list(batch)
        srcs, tgts = prepare_batch(batch)
        if not srcs:
            pbar.update(batch_size)
            continue
        try:
            inputs = tokenizer(srcs, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(device)
            with torch.no_grad():
                translated = model.generate(**inputs, max_length=MAX_LEN)
            hyps = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
            hypotheses.extend(hyps)
            references.extend([[t] for t in tgts])
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("   ❗ OOM, limpando cache…")
                torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                raise
        pbar.update(batch_size)
    pbar.close()
    bleu_score = bleu_metric.compute(predictions=hypotheses, references=references)["bleu"] * 100
    chrf_score = chrf_metric.compute(predictions=hypotheses, references=references)["score"]
    elapsed = time.time() - t_start
    return bleu_score, chrf_score, elapsed

# -------- Execução --------
print("[1/4] Salvando cabeçalho do CSV…")
save_header_if_needed(OUTPUT_FILE, [
    "Dataset","Tamanho Real","Modelo","Device","Batch Size","BLEU","chr-F","Tempo (s)"
])

print("[2/4] Carregando datasets…")
DATASETS = {}
DATASET_REAL_SIZE = {}
for name, loader, total_fn in DATASETS_INFO:
    try:
        ds = loader()
        DATASETS[name] = ds
        DATASET_REAL_SIZE[name] = total_fn()
        print(f"   ✅ {name} carregado ({len(ds)} amostras) | colunas: {ds.column_names}")
        try:
            print("   ↪ exemplo:", {k: ds[0][k] for k in ds.column_names})
        except Exception:
            pass
    except Exception as e:
        print(f"   ❌ Erro ao carregar {name}: {e}")

print("[3/4] Carregando modelo fine-tunado…")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(device)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
print(f"   ✅ Modelo carregado de {MODEL_DIR}")

print("[4/4] Iniciando avaliações…")
for dataset_name, ds in DATASETS.items():
    bs = BATCH_SIZE_BY_DATASET.get(dataset_name, 1)
    bleu_score, chrf_score, elapsed_time = compute_metrics(ds, model, tokenizer, bs, dataset_name)
    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            dataset_name,
            DATASET_REAL_SIZE.get(dataset_name, ""),
            os.path.basename(os.path.abspath(MODEL_DIR)),
            "cuda",
            bs,
            f"{bleu_score:.2f}",
            f"{chrf_score:.4f}",
            f"{elapsed_time:.2f}"
        ])
    print(f"   ✅ {dataset_name}: BLEU={bleu_score:.2f} | chrF={chrf_score:.4f} | Tempo={elapsed_time:.2f}s")

print("🏁 Avaliação concluída! Resultados salvos em", OUTPUT_FILE)

# -*- coding: utf-8 -*-
"""
Avaliacao do modelo quickmt/quickmt-en-pt (CTranslate2).

Requer: pip install quickmt huggingface_hub datasets evaluate tqdm unbabel-comet bert-score

Uso: python evaluate_quickmt.py [--full]
     --full: apaga CSV anterior e roda do zero

Gera resultados em evaluation_results/resultados_quickmt.csv
(mesmo formato CSV que models-test.py, com COMET e BERTScore F1).
"""
import os
import sys
import csv
import time
import gc
import argparse
from collections import OrderedDict
from datasets import load_dataset
from tqdm import tqdm
import evaluate

try:
    from quickmt import Translator
    from huggingface_hub import snapshot_download
except ImportError:
    print("Instale as dependencias: pip install quickmt huggingface_hub")
    print("   Ou: git clone https://github.com/quickmt/quickmt.git && pip install ./quickmt/")
    sys.exit(1)

# ==================== Config ====================
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    import torch
    DEVICE = "cpu"

MODEL_HF_ID = "quickmt/quickmt-en-pt"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_results")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "resultados_quickmt.csv")
BEAM_SIZE = 5
BATCH_SIZE = 32

MAX_SAMPLES_WMT24PP = 5000
MAX_SAMPLES_PARACRAWL = 5_000
MAX_SAMPLES_FLORES = 1012
MAX_SAMPLES_OPUS100 = 5_000
MAX_SAMPLES = None  # Override unico para teste rapido

CSV_HEADER = [
    "Dataset", "Tamanho Real", "Modelo", "Device", "Batch Size", "Tamanho Dataset (usado)",
    "Exemplos do Dataset", "Total de Sentencas", "Total de Palavras", "Media Palavras por Sentenca",
    "Tempo Total", "Tempo por Sentenca (s)", "Palavras por Segundo",
    "Uso de Memoria (RAM)", "Memoria GPU Alocada (MB)", "Memoria GPU Reservada (MB)",
    "BLEU", "chr-F", "COMET", "BERTScore F1", "Erro"
]

bleu_metric = evaluate.load("bleu")
chrf_metric = evaluate.load("chrf")

# ==================== COMET / BERTScore ====================
_comet_model = None
_use_comet = None
_use_bertscore = None


def _get_comet():
    global _comet_model, _use_comet
    if _use_comet is None:
        try:
            import comet
            from comet.models import load_from_checkpoint
            path = comet.download_model("Unbabel/wmt22-comet-da")
            _comet_model = load_from_checkpoint(path)
            _comet_model.eval()
            _use_comet = True
            print("[OK] COMET carregado (unbabel-comet).")
        except Exception as e:
            _comet_model = None
            _use_comet = False
            print(f"[AVISO] COMET nao disponivel: {e}")
    return _comet_model if _use_comet else None


def _bertscore_available():
    global _use_bertscore
    if _use_bertscore is None:
        try:
            import bert_score
            _use_bertscore = True
        except Exception:
            _use_bertscore = False
            print("[AVISO] BERTScore nao disponivel (pip install bert-score).")
    return _use_bertscore


def compute_neural_metrics(sources, hypotheses, references):
    """Calcula COMET e BERTScore F1. Retorna (comet_score, bertscore_f1)."""
    comet_score = None
    bertscore_f1 = None

    # COMET
    comet_model = _get_comet()
    if comet_model and sources and hypotheses and references:
        try:
            data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hypotheses, references)]
            gpus = 1 if DEVICE == "cuda" else 0
            output = comet_model.predict(data, gpus=gpus, progress_bar=False, num_workers=0)
            comet_score = float(output.system_score)
        except Exception as e:
            print(f"[AVISO] COMET falhou: {e}")
        finally:
            if comet_model is not None:
                try:
                    comet_model.cpu()
                except Exception:
                    pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # BERTScore
    if _bertscore_available() and hypotheses and references:
        try:
            from bert_score import score
            P, R, F = score(hypotheses, references, lang="pt", verbose=False, device=DEVICE)
            bertscore_f1 = float(F.mean())
        except Exception as e:
            print(f"[AVISO] BERTScore falhou: {e}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return comet_score, bertscore_f1


# ==================== Datasets ====================
def load_wmt24pp():
    ds = load_dataset("google/wmt24pp", "en-pt_BR", split="train")
    if MAX_SAMPLES_WMT24PP is not None and len(ds) > MAX_SAMPLES_WMT24PP:
        ds = ds.select(range(MAX_SAMPLES_WMT24PP))
    return ds


def load_paracrawl():
    n = MAX_SAMPLES_PARACRAWL or 5_000
    return load_dataset("para_crawl", "enpt", split=f"train[:{n}]", trust_remote_code=True)


def load_flores():
    ds = load_dataset("facebook/flores", "eng_Latn-por_Latn", split="devtest", trust_remote_code=True)
    if MAX_SAMPLES_FLORES is not None and len(ds) > MAX_SAMPLES_FLORES:
        ds = ds.select(range(MAX_SAMPLES_FLORES))
    return ds


def load_opus100():
    ds = load_dataset("opus100", "en-pt", split="test", trust_remote_code=True)
    if MAX_SAMPLES_OPUS100 is not None and len(ds) > MAX_SAMPLES_OPUS100:
        ds = ds.select(range(MAX_SAMPLES_OPUS100))
    return ds


def get_src_tgt_list(dataset, dataset_name):
    """Retorna listas (sources, references) para o dataset."""
    srcs, tgts = [], []
    for i in range(len(dataset)):
        ex = dataset[i]
        if dataset_name == "wmt24pp":
            srcs.append(ex.get("source", ""))
            tgts.append(ex.get("target", ""))
        elif dataset_name == "paracrawl":
            if "translation" in ex:
                srcs.append(ex["translation"].get("en", ""))
                tgts.append(ex["translation"].get("pt", ""))
            else:
                srcs.append(ex.get("source", ""))
                tgts.append(ex.get("target", ""))
        elif dataset_name == "flores":
            srcs.append(ex.get("sentence_eng_Latn", ""))
            tgts.append(ex.get("sentence_por_Latn", ""))
        elif dataset_name == "opus100":
            if "translation" in ex:
                srcs.append(ex["translation"].get("en", ""))
                tgts.append(ex["translation"].get("pt", ""))
            else:
                srcs.append(ex.get("en", ""))
                tgts.append(ex.get("pt", ""))
    return srcs, tgts


DATASETS_CONFIG = [
    ("wmt24pp", load_wmt24pp),
    ("paracrawl", load_paracrawl),
    ("flores", load_flores),
    ("opus100", load_opus100),
]


def save_header_if_needed(filename, header):
    if not os.path.exists(filename):
        with open(filename, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)


def already_done(filename, dataset_name, model_name):
    """Verifica se combinacao (dataset, modelo) ja esta no CSV."""
    if not os.path.exists(filename):
        return False
    try:
        with open(filename, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("Dataset", "").strip() == dataset_name and r.get("Modelo", "").strip() == model_name:
                    if not (r.get("Erro") or "").strip():
                        return True
    except Exception:
        pass
    return False


# ==================== Main ====================
def main():
    parser = argparse.ArgumentParser(description="Avaliacao do QuickMT (CTranslate2)")
    parser.add_argument("--full", action="store_true", help="Apaga CSV anterior e roda do zero")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.full and os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        print(f"[OK] CSV anterior removido: {OUTPUT_FILE}")

    save_header_if_needed(OUTPUT_FILE, CSV_HEADER)

    print(f"Baixando modelo {MODEL_HF_ID} (formato CTranslate2)...")
    model_path = snapshot_download(MODEL_HF_ID, ignore_patterns="eole-model/*")
    print(f"Modelo em: {model_path}")
    print(f"Dispositivo: {DEVICE}")
    t = Translator(model_path, device=DEVICE)

    for dataset_name, loader_fn in DATASETS_CONFIG:
        if not args.full and already_done(OUTPUT_FILE, dataset_name, MODEL_HF_ID):
            print(f"\n[SKIP] {dataset_name} x {MODEL_HF_ID} ja avaliado. Use --full para refazer.")
            continue

        print(f"\nDataset: {dataset_name}")
        try:
            ds = loader_fn()
        except Exception as e:
            print(f"Erro ao carregar {dataset_name}: {e}")
            continue

        srcs, tgts = get_src_tgt_list(ds, dataset_name)
        tamanho_real = len(srcs)
        if MAX_SAMPLES is not None:
            srcs, tgts = srcs[:MAX_SAMPLES], tgts[:MAX_SAMPLES]
        gc.collect()

        num_sentences = len(srcs)
        num_words = sum(len(s.split()) for s in tgts)
        media_palavras = num_words / num_sentences if num_sentences else 0
        exemplos_str = f"EN: {srcs[0][:80]}... || PT: {tgts[0][:80]}..." if srcs else ""

        hypotheses = []
        t0 = time.time()
        for start in tqdm(range(0, num_sentences, BATCH_SIZE), desc=f"quickmt {dataset_name}", ncols=100):
            batch_srcs = srcs[start:start + BATCH_SIZE]
            try:
                out = t(batch_srcs, beam_size=BEAM_SIZE)
                if isinstance(out, list):
                    hypotheses.extend(out)
                else:
                    hypotheses.append(str(out))
            except Exception as e:
                print(f"   Erro no batch {start}: {e}")
                hypotheses.extend([""] * len(batch_srcs))
        elapsed = time.time() - t0

        if len(hypotheses) != num_sentences:
            hypotheses = hypotheses[:num_sentences]

        references = [[t_] for t_ in tgts]
        bleu_score = bleu_metric.compute(predictions=hypotheses, references=references)["bleu"] * 100
        chrf_score = chrf_metric.compute(predictions=hypotheses, references=references)["score"]
        tempo_por_sent = elapsed / num_sentences if num_sentences else 0
        palavras_por_seg = num_words / elapsed if elapsed > 0 else 0

        # COMET + BERTScore
        refs_flat = [r[0] if isinstance(r, list) else r for r in references]
        print("[INFO] Calculando COMET e BERTScore...")
        comet_score, bertscore_f1 = compute_neural_metrics(srcs, hypotheses, refs_flat)

        row = [
            dataset_name, tamanho_real, MODEL_HF_ID, DEVICE, BATCH_SIZE, num_sentences,
            exemplos_str, num_sentences, num_words, f"{media_palavras:.2f}",
            f"{elapsed:.2f}s", f"{tempo_por_sent:.4f}", f"{palavras_por_seg:.2f}",
            "", "", "",  # RAM/GPU nao medidos
            f"{bleu_score:.2f}", f"{chrf_score:.5f}",
            f"{comet_score:.4f}" if comet_score is not None else "",
            f"{bertscore_f1:.4f}" if bertscore_f1 is not None else "",
            ""
        ]
        with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

        comet_str = f"COMET={comet_score:.4f}" if comet_score is not None else "COMET=N/A"
        bs_str = f"BERTScore={bertscore_f1:.4f}" if bertscore_f1 is not None else "BERTScore=N/A"
        print(f"   BLEU={bleu_score:.2f} chr-F={chrf_score:.5f} {comet_str} {bs_str} Tempo={elapsed:.2f}s")
        del srcs, tgts, hypotheses, references
        gc.collect()

    print(f"\nResultados salvos em {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

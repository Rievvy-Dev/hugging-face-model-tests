# -*- coding: utf-8 -*-
"""Métricas de avaliação: BLEU, chr-F, COMET e BERTScore F1."""
import gc
import time
import torch
from tqdm import tqdm
import evaluate

from . import config
from .datasets import extract_texts

bleu_metric = evaluate.load("bleu")
chrf_metric = evaluate.load("chrf")

# COMET e BERTScore: uso direto das bibliotecas (unbabel-comet, bert-score)
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


def _comet_available():
    return _get_comet() is not None


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


def compute_metrics(dataset, model, tokenizer, batch_size, dataset_name, model_name, device=None):
    """Retorna (bleu_score, chrf_score, comet_score, bertscore_f1, elapsed, erro_msg)."""
    device = device or config.device
    model.eval()
    references, hypotheses, sources = [], [], []
    num_examples = len(dataset)
    erro_msg = ""
    t_start = time.time()
    pbar = tqdm(total=num_examples, desc=f"Processando ({dataset_name}, batch={batch_size})", ncols=100)

    is_nllb = "nllb" in model_name.lower()
    is_m2m = "m2m100" in model_name.lower()
    forced_bos_token_id = None
    if is_nllb:
        tokenizer.src_lang = "eng_Latn"
        tokenizer.tgt_lang = "por_Latn"
        forced_bos_token_id = tokenizer.convert_tokens_to_ids("por_Latn")
    elif is_m2m:
        tokenizer.src_lang = "en"
        try:
            forced_bos_token_id = tokenizer.get_lang_id("pt")
        except Exception:
            forced_bos_token_id = tokenizer.convert_tokens_to_ids("pt")

    for start_idx in range(0, num_examples, batch_size):
        batch = dataset[start_idx:start_idx + batch_size]
        if isinstance(batch, dict):
            batch = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]
        elif hasattr(batch, "__iter__") and not isinstance(batch, (list, tuple)):
            batch = [dict(row) for row in batch]
        if not batch:
            continue

        srcs, tgts = extract_texts(batch, dataset_name)
        if not srcs:
            erro_msg += f" | Estrutura inesperada em idx {start_idx}"
            pbar.update(batch_size)
            continue

        try:
            inputs = tokenizer(srcs, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                if is_nllb or is_m2m:
                    out = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id)
                else:
                    out = model.generate(**inputs)
            batch_hyps = [str(tokenizer.decode(t, skip_special_tokens=True)).strip() or " " for t in out]
            hypotheses.extend(batch_hyps)
            references.extend([[str(t).strip() or " "] for t in tgts])
            sources.extend([str(s).strip() or " " for s in srcs])
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[!] OOM em idx {start_idx}, pulando batch...")
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

    # Liberar VRAM antes de carregar COMET e BERTScore (RTX 2050 = 4GB, não cabe tudo junto)
    print("[INFO] Liberando modelo de tradução da GPU para calcular COMET/BERTScore...")
    model.cpu()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    comet_score = None
    comet_model = _get_comet()
    if comet_model and sources and hypotheses and references:
        try:
            refs_flat = [r[0] if isinstance(r, list) else r for r in references]
            data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hypotheses, refs_flat)]
            gpus = 1 if (device and "cuda" in str(device)) else 0
            output = comet_model.predict(data, gpus=gpus, progress_bar=False, num_workers=0)
            comet_score = float(output.system_score)
        except Exception as e:
            print(f"[AVISO] COMET falhou: {e}")
        finally:
            # Liberar COMET da GPU antes do BERTScore
            if comet_model is not None:
                try:
                    comet_model.cpu()
                except Exception:
                    pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    bertscore_f1 = None
    if _bertscore_available() and hypotheses and references:
        try:
            from bert_score import score
            refs_flat = [r[0] if isinstance(r, list) else r for r in references]
            P, R, F = score(hypotheses, refs_flat, lang="pt", verbose=False, device=device)
            bertscore_f1 = float(F.mean())
        except Exception as e:
            print(f"[AVISO] BERTScore falhou: {e}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Mover modelo de tradução de volta para GPU para o próximo dataset
    model.to(device)

    return bleu_score, chrf_score, comet_score, bertscore_f1, elapsed, erro_msg

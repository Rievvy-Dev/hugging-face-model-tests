# -*- coding: utf-8 -*-
"""Carregamento e helpers dos datasets de avaliação."""
from datasets import load_dataset

from . import config


def load_wmt24pp():
    ds = load_dataset("google/wmt24pp", "en-pt_BR", split="train")
    if not config.USE_FULL_DATASETS:
        ds = ds.shuffle(seed=42).select(range(min(998, len(ds))))
    elif config.MAX_SAMPLES_WMT24PP is not None and len(ds) > config.MAX_SAMPLES_WMT24PP:
        ds = ds.select(range(config.MAX_SAMPLES_WMT24PP))
    return ds


def load_paracrawl():
    n = (config.MAX_SAMPLES_PARACRAWL if config.USE_FULL_DATASETS else config.PARACRAWL_N) or 20_000
    try:
        return load_dataset("para_crawl", "enpt", split=f"train[:{n}]")
    except Exception:
        ds = load_dataset("opus100", "en-pt", split=f"train[:{n}]")
        return ds.shuffle(seed=42) if len(ds) > n else ds


def load_flores():
    try:
        ds = load_dataset("facebook/flores", "eng_Latn-por_Latn", split="devtest")
    except Exception:
        ds = load_dataset("opus100", "en-pt", split="train[:1012]").shuffle(seed=42)
    if config.MAX_SAMPLES_FLORES is not None and len(ds) > config.MAX_SAMPLES_FLORES:
        ds = ds.select(range(config.MAX_SAMPLES_FLORES))
    return ds


def load_opus100():
    n = config.MAX_SAMPLES_OPUS100 or 10_000
    ds = load_dataset("opus100", "en-pt", split=f"train[:{n}]")
    if len(ds) > n:
        ds = ds.shuffle(seed=42).select(range(n))
    return ds


DATASETS_INFO = [
    ("wmt24pp",  load_wmt24pp),
    ("paracrawl", load_paracrawl),
    ("flores",   load_flores),
    ("opus100",  load_opus100),
]


def extract_texts(batch, dataset_name):
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
    elif dataset_name == "flores":
        if batch and "translation" in batch[0]:
            srcs = [ex["translation"].get("en", ex["translation"].get("source", "")) for ex in batch]
            tgts = [ex["translation"].get("pt", ex["translation"].get("target", "")) for ex in batch]
        else:
            srcs = [ex.get("sentence_eng_Latn", ex.get("sentence_eng", "")) for ex in batch]
            tgts = [ex.get("sentence_por_Latn", ex.get("sentence_por", "")) for ex in batch]
    elif dataset_name == "opus100":
        if batch and "translation" in batch[0]:
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
    elif dataset_name == "flores":
        if dataset and "translation" in dataset[0]:
            textos = [ex["translation"].get("pt", ex["translation"].get("target", "")) for ex in dataset]
        else:
            textos = [ex.get("sentence_por_Latn", ex.get("sentence_por", "")) for ex in dataset]
    elif dataset_name == "opus100":
        if dataset and "translation" in dataset[0]:
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
        elif dataset_name == "flores":
            if "translation" in s:
                exemplos.append(f"EN: {s['translation'].get('en','')}\nPT: {s['translation'].get('pt','')}")
            else:
                exemplos.append(f"EN: {s.get('sentence_eng_Latn','')}\nPT: {s.get('sentence_por_Latn','')}")
        elif dataset_name == "opus100":
            if "translation" in s:
                exemplos.append(f"EN: {s['translation'].get('en','')}\nPT: {s['translation'].get('pt','')}")
            else:
                exemplos.append(f"EN: {s.get('source','')}\nPT: {s.get('target','')}")
    return " || ".join(exemplos)

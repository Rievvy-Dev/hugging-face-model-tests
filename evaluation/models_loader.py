# -*- coding: utf-8 -*-
"""Carregamento de modelos (Helsinki, M2M100, Auto)."""
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    MarianMTModel,
    MarianTokenizer,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
)

from . import config


def load_single_model(model_name, device=None):
    """Carrega um modelo e tokenizer. Helsinki usa MarianMT; M2M100 usa M2M100*; demais Auto*."""
    device = device or config.device
    # QuickMT / CTranslate2 models: use quickmt.Translator (no HF tokenizer)
    if "quickmt" in (model_name or "").lower():
        from huggingface_hub import snapshot_download
        # Avoid creating symlinks on Windows without privileges
        import os as _os
        _os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
        model_path = snapshot_download(model_name, ignore_patterns="eole-model/*")
        # Prefer quickmt if available, otherwise fallback to ctranslate2.Translator
        try:
            from quickmt import Translator as QuickMTTranslator
            translator = QuickMTTranslator(model_path, device=device)
            return translator, None
        except Exception:
            try:
                from ctranslate2 import Translator as CTranslate2Translator
                # ctranslate2 expects device as 'cpu' or 'cuda'
                ct_device = "cuda" if (device and "cuda" in str(device)) else "cpu"
                translator = CTranslate2Translator(model_path, device=ct_device)
                return translator, None
            except Exception as e:
                raise RuntimeError(f"Erro ao carregar QuickMT / CTranslate2 model '{model_name}': {e}")
    if "Helsinki-NLP" in model_name or "opus-mt" in model_name:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(device)
    elif "m2m100" in model_name.lower():
        tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        model = M2M100ForConditionalGeneration.from_pretrained(model_name).to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return model, tokenizer

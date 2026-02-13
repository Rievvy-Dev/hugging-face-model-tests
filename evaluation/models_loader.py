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

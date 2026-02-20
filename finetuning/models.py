# -*- coding: utf-8 -*-
"""Carregamento de modelos e tokenizadores."""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from . import config


def load_model(model_name, pretrained=True):
    """
    Carrega modelo de tradução.
    
    Args:
        model_name: Nome do modelo ('helsinki', 'm2m100') ou path completo
        pretrained: Se True, carrega modelo pré-treinado da HF
    
    Returns:
        model: Modelo pronto para uso
    """
    # Se receber nome curto, traduzir para path completo
    if model_name in config.MODELS:
        model_path = config.MODELS[model_name]
    else:
        model_path = model_name
    
    print(f"   📦 Carregando modelo: {model_path}")
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model = model.to(config.device)
    model.eval()
    
    return model


def load_tokenizer(model_name):
    """
    Carrega tokenizador.
    
    Args:
        model_name: Nome do modelo ('helsinki', 'm2m100') ou path completo
    
    Returns:
        tokenizer: Tokenizador pronto para uso
    """
    # Se receber nome curto, traduzir para path completo
    if model_name in config.MODELS:
        model_path = config.MODELS[model_name]
    else:
        model_path = model_name
    
    print(f"   🔤 Carregando tokenizador: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return tokenizer


def load_model_and_tokenizer(model_name, pretrained=True):
    """
    Carrega modelo e tokenizador juntos.
    
    Args:
        model_name: Nome do modelo
        pretrained: Se True, carrega pré-treinado
    
    Returns:
        tuple: (model, tokenizer)
    """
    model = load_model(model_name, pretrained=pretrained)
    tokenizer = load_tokenizer(model_name)
    
    return model, tokenizer


def model_require_lang_code(model_name):
    """
    Verifica se modelo requer código de linguagem.
    
    M2M100 requer: __pt_BR__ na entrada
    MarianMT não requer.
    
    Returns:
        (bool, str): (requires, lang_code)
    """
    if "m2m100" in model_name.lower():
        return True, "__pt_BR__"
    return False, None

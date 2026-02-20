# -*- coding: utf-8 -*-
"""Cálculo de métricas de tradução."""
import math
import torch
from sacrebleu import BLEU, CHRF
from . import config


def calculate_bleu(predictions, references, lowercase=False):
    """
    Calcula BLEU score.
    
    Args:
        predictions: list de strings
        references: list de strings
        lowercase: Se True, calcula case-insensitive
    
    Returns:
        float: BLEU score (0-100)
    """
    bleu = BLEU(lowercase=lowercase)
    
    # sacrebleu espera: lang_pair='en-pt_BR'
    bleu_score = bleu.corpus_score(predictions, [references])
    
    return bleu_score.score


def calculate_chrf(predictions, references, lowercase=False):
    """
    Calcula chr-F score (character F-score).
    
    Args:
        predictions: list de strings
        references: list de strings
        lowercase: Se True, calcula case-insensitive
    
    Returns:
        float: chr-F score (0-100)
    """
    chrf = CHRF(lowercase=lowercase)
    chrf_score = chrf.corpus_score(predictions, [references])
    
    return chrf_score.score


def evaluate_batch(model, tokenizer, sources, references, model_name, batch_size=config.DEFAULT_EVAL_BATCH_SIZE):
    """
    Avalia batch de exemplos.
    
    Args:
        model: Modelo de tradução
        tokenizer: Tokenizador
        sources: list de strings em inglês
        references: list de strings em português
        model_name: Nome do modelo (ex: 'helsinki', 'm2m100')
        batch_size: Tamanho do batch
    
    Returns:
        dict: {bleu, chrf, comet, bertscore}
    """
    
    requires_lang, lang_code = config_model_lang_code(model_name)
    
    predictions = []
    
    # Traduzir em batches
    with torch.no_grad():
        for i in range(0, len(sources), batch_size):
            batch_sources = sources[i:i+batch_size]
            
            # Adicionar código de linguagem se necessário (M2M100)
            if requires_lang:
                batch_inputs = [f"{lang_code}{src}" for src in batch_sources]
            else:
                batch_inputs = batch_sources
            
            # Tokenizar
            inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(config.device) for k, v in inputs.items()}
            
            # Gerar
            generated = model.generate(**inputs, num_beams=5, max_length=256)
            
            # Decodificar
            batch_preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
            predictions.extend(batch_preds)
    
    # Calcular métricas
    bleu = calculate_bleu(predictions, references)
    chrf = calculate_chrf(predictions, references)
    
    return {
        "model": model_name,
        "bleu": bleu,
        "chrf": chrf,
        "comet": None,  # Implementar depois se necessário
        "bertscore": None,  # Implementar depois se necessário
    }


def config_model_lang_code(model_name):
    """Helper para verificar se modelo requer código de idioma."""
    from .models import model_require_lang_code
    return model_require_lang_code(model_name)

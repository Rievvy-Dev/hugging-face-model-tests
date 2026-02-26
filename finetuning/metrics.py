# -*- coding: utf-8 -*-
"""Cálculo de métricas de tradução."""
import math
import torch
from sacrebleu import BLEU, CHRF
from . import config

# Cache para modelos COMET e BERTScore
_comet_model = None
_bertscore_available = None


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


def calculate_comet(predictions, references, sources):
    """
    Calcula COMET score (da Unbabel).
    
    Args:
        predictions: list de strings (traduções)
        references: list de strings (referência)
        sources: list de strings (original em inglês)
    
    Returns:
        float: COMET score (0-1)
    """
    try:
        from comet import download_model, load_from_checkpoint
        
        global _comet_model
        if _comet_model is None:
            print("   📥 Baixando modelo COMET...")
            model_path = download_model("Unbabel/wmt22-comet-da")
            _comet_model = load_from_checkpoint(model_path)
            _comet_model.eval()
        
        # Preparar dados no formato esperado
        data = [
            {"src": src, "mt": pred, "ref": ref}
            for src, pred, ref in zip(sources, predictions, references)
        ]
        
        print("   🔄 Calculando COMET...")
        output = _comet_model.predict(data, batch_size=2, gpus=1 if torch.cuda.is_available() else 0)
        
        return float(output.system_score)
    
    except Exception as e:
        print(f"   ⚠️  COMET falhou: {e}")
        return None


def calculate_bertscore(predictions, references):
    """
    Calcula BERTScore F1.
    
    Args:
        predictions: list de strings (traduções)
        references: list de strings (referência)
    
    Returns:
        float: BERTScore F1 médio (0-1)
    """
    try:
        from bert_score import score
        
        print("   🔄 Calculando BERTScore...")
        # Usar modelo multilíngue para português
        P, R, F1 = score(
            predictions, 
            references, 
            lang="pt",
            batch_size=2,
            device=config.device
        )
        
        return float(F1.mean())
    
    except Exception as e:
        print(f"   ⚠️  BERTScore falhou: {e}")
        return None

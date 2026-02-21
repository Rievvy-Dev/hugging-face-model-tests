#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para calcular apenas COMET e BERTScore (métricas neurais).
Usa um modelo já carregado para traduzir e calcula apenas as métricas neurais.

Uso:
  python compute_neural_metrics.py --model helsinki
  python compute_neural_metrics.py --model m2m100 --finetuned
"""
import os
import sys
import torch
from tqdm import tqdm

# Instalar se necessário: 
# pip install comet-ml unbabel-comet bert-score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from finetuning import config, data_utils


def load_model_and_tokenizer(model_name):
    """Carrega modelo e tokenizer."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    
    print(f"    Baixando {model_name}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model.to(config.device)
    model.eval()
    
    return model, tokenizer


def translate_batch(model, tokenizer, sources, batch_size=2):
    """Traduz batch de textos."""
    predictions = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(sources), batch_size), desc="     Traduzindo", unit="batch"):
            batch_sources = sources[i:i+batch_size]
            
            inputs = tokenizer(batch_sources, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(config.device) for k, v in inputs.items()}
            
            generated = model.generate(**inputs, num_beams=5, max_length=256)
            batch_preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
            predictions.extend(batch_preds)
    
    return predictions


def calculate_comet(predictions, references, sources, batch_size=4, device=None):
    """Calcula COMET."""
    print("     📊 Calculando COMET...")

    try:
        import comet
        from comet.models import load_from_checkpoint

        model_path = comet.download_model("Unbabel/wmt22-comet-da")
        comet_model = load_from_checkpoint(model_path)
        comet_model.eval()

        refs_flat = [r[0] if isinstance(r, list) else r for r in references]
        data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, predictions, refs_flat)]

        gpus = 1 if (device and "cuda" in str(device)) else 0
        output = comet_model.predict(data, batch_size=batch_size, gpus=gpus, progress_bar=False, num_workers=0)
        comet_score = float(output.system_score)

        print(f"        ✅ COMET: {comet_score:.4f}")
        return comet_score

    except Exception as e:
        print(f"        ❌ Erro COMET: {e}")
        return None


def calculate_bertscore(predictions, references, batch_size=2, device=None):
    """
    Calcula BERTScore.
    
    Args:
        predictions: list de strings
        references: list de strings  
        batch_size: Batch size
    """
    print("     📊 Calculando BERTScore...")
    
    try:
        from bert_score import score
        
        ref_sample = [r[0] if isinstance(r, list) else r for r in references]
        
        precision, recall, f1 = score(
            predictions,
            ref_sample,
            lang="pt",
            batch_size=batch_size,
            device=device or "cpu",
            verbose=False
        )
        
        f1_mean = f1.mean().item()
        
        print(f"        ✅ BERTScore F1: {f1_mean:.4f}")
        return f1_mean
        
    except Exception as e:
        print(f"        ❌ Erro BERTScore: {e}")
        return None


def main():
    """Calcula COMET e BERTScore para um modelo."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Calcular COMET e BERTScore")
    parser.add_argument(
        "--model",
        type=str,
        choices=list(config.MODELS.keys()),
        required=True,
        help="Modelo a avaliar (helsinki ou m2m100)"
    )
    parser.add_argument(
        "--finetuned",
        action="store_true",
        help="Usar modelo fine-tuned em vez do base"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size para tradução"
    )
    parser.add_argument(
        "--metric_batch_size",
        type=int,
        default=2,
        help="Batch size para métricas COMET/BERTScore"
    )
    
    args = parser.parse_args()
    
    # Carregar dados de teste
    print(f"\n📂 Carregando dados de teste...")
    test_samples = data_utils.load_scielo_csv(config.SCIELO_TEST_CSV)
    sources = [s.get("abstract_en", "").strip() for s in test_samples]
    references = [s.get("abstract_pt", "").strip() for s in test_samples]
    
    print(f"\n{'='*80}")
    print(f"  🔬 COMET + BERTScore para {args.model.upper()}")
    if args.finetuned:
        print(f"    (MODELO FINE-TUNED)")
    print(f"{'='*80}\n")
    print(f"  📊 Avaliando {len(sources):,} exemplos\n")
    
    # Carregar modelo
    if args.finetuned:
        model_path = f"./models/finetuned-scielo/{args.model}"
        if not os.path.exists(model_path):
            print(f"  ❌ Modelo fine-tuned não encontrado: {model_path}\n")
            return
        model_id = model_path
        print(f"  🤖 Carregando modelo fine-tuned...")
    else:
        model_id = config.MODELS[args.model]
        print(f"  🤖 Carregando modelo base...")
    
    model, tokenizer = load_model_and_tokenizer(model_id)
    
    # Traduzir
    print(f"\n  🔄 Traduzindo {len(sources):,} exemplos...\n")
    predictions = translate_batch(model, tokenizer, sources, batch_size=args.batch_size)
    
    # Liberar VRAM do modelo de traducao antes de COMET/BERTScore
    if hasattr(model, "cpu"):
        model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Calcular métricas neurais
    print(f"\n  📈 Calculando métricas neurais...\n")
    comet = calculate_comet(
        predictions,
        references,
        sources,
        batch_size=args.metric_batch_size,
        device=config.device,
    )
    bertscore = calculate_bertscore(
        predictions,
        references,
        batch_size=2,
        device=config.device,
    )
    
    print(f"\n{'='*80}")
    print(f"  ✅ RESULTADO")
    print(f"{'='*80}")
    if comet is not None:
        print(f"  COMET: {comet:.4f}")
    else:
        print(f"  COMET: ❌ Erro")
    if bertscore is not None:
        print(f"  BERTScore F1: {bertscore:.4f}")
    else:
        print(f"  BERTScore: ❌ Erro")
    print()
    
    # Limpar memória
    del model, tokenizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

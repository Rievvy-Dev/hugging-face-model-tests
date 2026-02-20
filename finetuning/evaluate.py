# -*- coding: utf-8 -*-
"""Avaliação de modelos (antes e depois do fine-tuning)."""
import torch
from tqdm import tqdm
from .models import load_model_and_tokenizer
from .data_utils import get_test_samples
from .metrics import calculate_bleu, calculate_chrf
from .io_utils import write_metrics_csv
from . import config


def translate_batch(model, tokenizer, sources, model_name, batch_size=config.DEFAULT_EVAL_BATCH_SIZE):
    """
    Traduz batch de exemplos.
    
    Args:
        model: Modelo de tradução
        tokenizer: Tokenizador
        sources: list de strings em inglês
        model_name: Nome do modelo (para detectar lang code)
        batch_size: Tamanho do batch
    
    Returns:
        list: Strings traduzidas
    """
    
    # M2M100 requer código de linguagem
    requires_lang = "m2m100" in model_name.lower()
    
    predictions = []
    
    # Criar barra de progresso
    num_batches = (len(sources) + batch_size - 1) // batch_size
    pbar = tqdm(total=num_batches, desc="Traduzindo", unit="batch")
    
    with torch.no_grad():
        for i in range(0, len(sources), batch_size):
            batch_sources = sources[i:i+batch_size]
            
            # Adicionar lang code se necessário
            if requires_lang:
                batch_inputs = [f"__pt_BR__{src}" for src in batch_sources]
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
            
            pbar.update(1)
    
    pbar.close()
    
    return predictions


def evaluate_models(model_names, test_csv=config.SCIELO_TEST_CSV, output_file=None):
    """
    Avalia múltiplos modelos em dataset de teste.
    
    Args:
        model_names: list de nomes ('helsinki', 'm2m100')
        test_csv: Path ao CSV de teste
        output_file: Path ao CSV de saída (default: BEFORE_METRICS_FILE)
    
    Returns:
        list: Dicts com métricas por modelo
    """
    
    if output_file is None:
        output_file = config.BEFORE_METRICS_FILE
    
    # Carregar amostras de teste
    test_samples = get_test_samples(test_csv)
    sources = [s.get("abstract_en", "").strip() for s in test_samples]
    references = [s.get("abstract_pt", "").strip() for s in test_samples]
    
    print(f"\n📚 Avaliando {len(sources):,} exemplos de teste\n")
    
    results = []
    
    # Barra de progresso para modelos
    for model_name in tqdm(model_names, desc="Avaliando modelos", unit="modelo"):
        print(f"\n  🔄 {model_name}...")
        
        try:
            # Carregar modelo
            model, tokenizer = load_model_and_tokenizer(model_name)
            
            # Traduzir
            predictions = translate_batch(model, tokenizer, sources, model_name, batch_size=config.DEFAULT_EVAL_BATCH_SIZE)
            
            # Calcular métricas
            bleu = calculate_bleu(predictions, references)
            chrf = calculate_chrf(predictions, references)
            
            result = {
                "model": model_name,
                "bleu": bleu,
                "chrf": chrf,
                "comet": None,
                "bertscore": None,
            }
            results.append(result)
            
            print(f"     ✅ BLEU: {bleu:.2f}, chr-F: {chrf:.2f}")
            
            # Liberar memória
            del model, tokenizer
            torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"     ❌ Erro: {e}")
            results.append({
                "model": model_name,
                "bleu": None,
                "chrf": None,
                "comet": None,
                "bertscore": None,
            })
    
    # Salvar
    write_metrics_csv(results, output_file)
    
    return results


def evaluate_before(test_csv=config.SCIELO_TEST_CSV, model_name=None):
    """
    Avalia modelos ORIGINAIS (antes do fine-tuning).
    
    Args:
        test_csv: Path ao CSV de teste
        model_name: Nome específico do modelo (None = todos)
    
    Returns:
        list: Métricas dos modelos base
    """
    print(f"\n{'='*80}")
    print(f"  PASSO 1: Avaliar Modelos ANTES do Fine-tuning")
    print(f"{'='*80}\n")
    
    if model_name:
        model_names = [model_name]
    else:
        model_names = list(config.MODELS.keys())
    
    return evaluate_models(
        model_names,
        test_csv=test_csv,
        output_file=config.BEFORE_METRICS_FILE
    )


def evaluate_after(test_csv=config.SCIELO_TEST_CSV, model_name=None):
    """
    Avalia modelos FINE-TUNED (depois do fine-tuning).
    
    Carrega modelos de ./models/finetuned-scielo/
    
    Args:
        test_csv: Path ao CSV de teste
        model_name: Nome específico do modelo (None = todos)
    
    Returns:
        list: Métricas dos modelos fine-tuned
    """
    print(f"\n{'='*80}")
    print(f"  PASSO 5: Avaliar Modelos DEPOIS do Fine-tuning")
    print(f"{'='*80}\n")
    
    # Carregar modelos fine-tuned
    model_names = []
    
    if model_name:
        # Modelo específico
        finetuned_path = f"./models/finetuned-scielo/{model_name}"
        import os
        if os.path.exists(finetuned_path):
            model_names.append(finetuned_path)
        else:
            print(f"⚠️  Modelo fine-tuned não encontrado: {finetuned_path}")
    else:
        # Todos os modelos
        for model_short in config.MODELS.keys():
            finetuned_path = f"./models/finetuned-scielo/{model_short}"
            import os
            if os.path.exists(finetuned_path):
                model_names.append(finetuned_path)
            else:
                print(f"⚠️  Modelo fine-tuned não encontrado: {finetuned_path}")
    
    if not model_names:
        print("❌ Nenhum modelo fine-tuned encontrado!")
        return []
    
    return evaluate_models(
        model_names,
        test_csv=test_csv,
        output_file=config.AFTER_METRICS_FILE
    )

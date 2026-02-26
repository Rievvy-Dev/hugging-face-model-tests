# -*- coding: utf-8 -*-
"""Avaliação de modelos (antes e depois do fine-tuning)."""
import gc
import json
import os
import torch
from tqdm import tqdm
from .models import load_model_and_tokenizer
from .data_utils import get_test_samples
from .metrics import calculate_bleu, calculate_chrf, calculate_comet, calculate_bertscore
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


def evaluate_models(model_names, test_csv=config.SCIELO_TEST_CSV, output_file=None, write_output=True):
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
            comet = calculate_comet(predictions, references, sources)
            bertscore = calculate_bertscore(predictions, references)
            
            result = {
                "model": model_name,
                "bleu": bleu,
                "chrf": chrf,
                "comet": comet,
                "bertscore": bertscore,
            }
            results.append(result)
            
            print(f"     ✅ BLEU: {bleu:.2f}, chr-F: {chrf:.2f}, COMET: {comet}, BERTScore: {bertscore}")
            
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
    if write_output:
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


def _format_epoch_tag(epoch_value):
    """Formata o identificador da epoca para uso em nome de arquivo."""
    if epoch_value is None:
        return "unknown"
    try:
        epoch_float = float(epoch_value)
        if abs(epoch_float - round(epoch_float)) < 1e-6:
            return str(int(round(epoch_float)))
        return str(epoch_float).replace(".", "p")
    except (TypeError, ValueError):
        return "unknown"


def _collect_checkpoint_infos(checkpoints_root, model_name=None):
    """Coleta informacoes de checkpoints para avaliacao."""
    checkpoint_infos = []

    if not os.path.exists(checkpoints_root):
        print(f"❌ Diretório de checkpoints nao encontrado: {checkpoints_root}")
        return checkpoint_infos

    model_dirs = []
    if model_name:
        model_dirs = [model_name]
    else:
        model_dirs = [d for d in os.listdir(checkpoints_root) if os.path.isdir(os.path.join(checkpoints_root, d))]

    for model_dir in model_dirs:
        model_root = os.path.join(checkpoints_root, model_dir)
        if not os.path.isdir(model_root):
            print(f"⚠️  Modelo ignorado (pasta nao encontrada): {model_root}")
            continue

        for entry in os.listdir(model_root):
            if not entry.startswith("checkpoint-"):
                continue
            checkpoint_path = os.path.join(model_root, entry)
            if not os.path.isdir(checkpoint_path):
                continue

            trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")
            epoch_value = None
            if os.path.exists(trainer_state_path):
                try:
                    with open(trainer_state_path, "r", encoding="utf-8") as f:
                        trainer_state = json.load(f)
                        epoch_value = trainer_state.get("epoch")
                except (OSError, json.JSONDecodeError) as e:
                    print(f"⚠️  Falha ao ler {trainer_state_path}: {e}")

            step_value = None
            try:
                step_value = int(entry.split("checkpoint-")[-1])
            except ValueError:
                step_value = None

            checkpoint_infos.append({
                "model": model_dir,
                "checkpoint": entry,
                "checkpoint_path": checkpoint_path,
                "epoch": epoch_value,
                "step": step_value,
            })

    checkpoint_infos.sort(key=lambda item: (
        item["model"],
        item["epoch"] if item["epoch"] is not None else 0,
        item["step"] if item["step"] is not None else 0,
    ))
    return checkpoint_infos


def evaluate_checkpoints(test_csv=config.SCIELO_TEST_CSV, model_name=None, checkpoints_root=None):
    """
    Avalia checkpoints de modelos fine-tuned por epoca.

    Args:
        test_csv: Path ao CSV de teste
        model_name: Nome especifico do modelo (None = todos)
        checkpoints_root: Diretorio raiz de checkpoints
    """
    checkpoints_root = checkpoints_root or config.FINETUNED_CHECKPOINTS_ROOT

    print(f"\n{'='*80}")
    print("  PASSO 5: Avaliar Checkpoints por Epoca")
    print(f"{'='*80}\n")
    print(f"📂 Raiz de checkpoints: {checkpoints_root}\n")

    checkpoint_infos = _collect_checkpoint_infos(checkpoints_root, model_name=model_name)
    if not checkpoint_infos:
        print("❌ Nenhum checkpoint encontrado para avaliacao.")
        return {}

    test_samples = get_test_samples(test_csv)
    sources = [s.get("abstract_en", "").strip() for s in test_samples]
    references = [s.get("abstract_pt", "").strip() for s in test_samples]
    print(f"📚 Avaliando {len(sources):,} exemplos de teste\n")

    results_by_epoch = {}

    for info in checkpoint_infos:
        epoch_tag = _format_epoch_tag(info["epoch"])
        output_file = config.AFTER_METRICS_EPOCH_TEMPLATE.format(epoch=epoch_tag)

        print(f"\n  🔄 {info['model']} | {info['checkpoint']} | epoca {info['epoch']}")
        model, tokenizer = load_model_and_tokenizer(info["checkpoint_path"])

        predictions = translate_batch(
            model,
            tokenizer,
            sources,
            info["model"],
            batch_size=config.DEFAULT_EVAL_BATCH_SIZE,
        )

        bleu = calculate_bleu(predictions, references)
        chrf = calculate_chrf(predictions, references)
        comet = calculate_comet(predictions, references, sources)
        bertscore = calculate_bertscore(predictions, references)

        result = {
            "model": info["model"],
            "checkpoint": info["checkpoint"],
            "epoch": info["epoch"],
            "bleu": bleu,
            "chrf": chrf,
            "comet": comet,
            "bertscore": bertscore,
        }
        results_by_epoch.setdefault(epoch_tag, []).append(result)

        write_metrics_csv(
            results_by_epoch[epoch_tag],
            output_file,
            fieldnames=["model", "checkpoint", "epoch", "bleu", "chrf", "comet", "bertscore"],
        )

        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    return results_by_epoch


def evaluate_after(test_csv=config.SCIELO_TEST_CSV, model_name=None, checkpoints_root=None):
    """
    Avalia modelos FINE-TUNED (depois do fine-tuning).
    
    Carrega modelos fine-tuned salvos por checkpoint.
    
    Args:
        test_csv: Path ao CSV de teste
        model_name: Nome específico do modelo (None = todos)
    
    Returns:
        list: Métricas dos modelos fine-tuned
    """
    return evaluate_checkpoints(
        test_csv=test_csv,
        model_name=model_name,
        checkpoints_root=checkpoints_root,
    )

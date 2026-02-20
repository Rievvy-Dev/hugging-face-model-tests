# -*- coding: utf-8 -*-
"""
Script principal: carrega datasets, avalia modelo por modelo, grava CSV agregado e um CSV por modelo.
Mantém a mesma lógica do models-test.py original.
Uso: python -m evaluation.run [--resume]
"""
import argparse
import csv
import gc
import os
import torch
import psutil

from . import config
from .config import get_output_dir, get_batch_sizes, get_csv_header, model_name_to_slug
from .datasets import DATASETS_INFO, get_dataset_stats, get_first_examples
from .io_utils import save_header_if_needed, sanitize_csv_cell, print_table
from .metrics import compute_metrics
from .models_loader import load_single_model


def evaluate_model_on_dataset(
    dataset_name, dataset, total_real, model_name, model, tokenizer, batch_size,
    output_dir, file_all, header, batch_size_map, done_pairs
):
    """Avalia um modelo em um dataset e grava no CSV agregado e no CSV do modelo."""
    if (model_name, dataset_name) in done_pairs:
        print(f"[PULADO] {model_name} x {dataset_name} já avaliado (--resume).")
        return

    device = config.device
    print(f"\n[Dataset] Avaliando dataset: {dataset_name} - modelo: {model_name}")
    exemplos_do_dataset = get_first_examples(dataset, dataset_name, 2)
    num_sentences, num_words, media_palavras = get_dataset_stats(dataset, dataset_name)
    print(f"  - Tamanho real do dataset: {total_real}")
    print(f"  - Total de sentenças (amostradas): {num_sentences}")
    print(f"  - Total de palavras: {num_words}")
    print(f"  - Média de palavras por sentença: {media_palavras:.2f}")
    batch_size = batch_size or 1
    print(f"  - Usando batch size: {batch_size}")

    process = psutil.Process()
    memory_before = process.memory_info().rss
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    try:
        print(f"[OK] Modelo carregado em: {device}")
        bleu_score, chrf_score, comet_score, bertscore_f1, elapsed_time, erro_msg = compute_metrics(
            dataset, model, tokenizer, batch_size, dataset_name, model_name, device
        )

        memory_after = process.memory_info().rss
        gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
        gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 ** 2) if torch.cuda.is_available() else 0

        # Use absolute memory after (MB) to avoid negative deltas caused by GC
        memory_after_mb = memory_after / (1024 * 1024)
        tempo_por_sentenca = elapsed_time / num_sentences if num_sentences else 0.0
        palavras_por_segundo = (num_words / elapsed_time) if elapsed_time > 0 else 0.0

        comet_str = f"{comet_score:.4f}" if comet_score is not None else ""
        bs_str = f"{bertscore_f1:.4f}" if bertscore_f1 is not None else ""
        data = {
            "Metric": header,
            "Valor": [
                dataset_name, total_real, model_name, device, batch_size, len(dataset),
                exemplos_do_dataset, num_sentences, num_words, f"{media_palavras:.2f}",
                f"{elapsed_time:.2f}s", f"{tempo_por_sentenca:.4f}", f"{palavras_por_segundo:.2f}",
                f"{memory_after_mb:.2f} MB",
                f"{gpu_memory_allocated:.2f}", f"{gpu_memory_reserved:.2f}",
                f"{bleu_score:.2f}", f"{chrf_score:.5f}", comet_str, bs_str, erro_msg
            ]
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        erro_msg = f"{type(e).__name__}: {e}"
        data = {
            "Metric": header,
            "Valor": [
                dataset_name, "", model_name, device, batch_size, "",
                exemplos_do_dataset, "", "", "", "", "", "",
                "", "", "", "", "", "", "", erro_msg
            ]
        }
        print(f"[ERRO] Erro ao avaliar modelo {model_name} no dataset {dataset_name}: {erro_msg}")

    print_table(data)
    row = [sanitize_csv_cell(v, 1500) for v in data["Valor"]]

    # Gravar no CSV agregado
    path_all = os.path.join(output_dir, file_all)
    save_header_if_needed(path_all, header)
    with open(path_all, mode="a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)

    # Gravar no CSV do modelo
    slug = model_name_to_slug(model_name)
    path_model = os.path.join(output_dir, f"{slug}.csv")
    save_header_if_needed(path_model, header)
    with open(path_model, mode="a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_done_pairs(output_dir, file_all):
    """Retorna set de (model_name, dataset_name) já presentes no CSV agregado."""
    path = os.path.join(output_dir, file_all)
    if not os.path.exists(path):
        return set()
    done = set()
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                m, d = r.get("Modelo", "").strip(), r.get("Dataset", "").strip()
                if m and d:
                    done.add((m, d))
    except Exception:
        pass
    return done


def main():
    parser = argparse.ArgumentParser(description="Avaliação de modelos de tradução (modelo por modelo).")
    parser.add_argument("--full", action="store_true", help="Rodar todas as combinações (ignorar o que já está salvo)")
    args = parser.parse_args()

    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    file_all = config.OUTPUT_FILE_ALL
    header = get_csv_header()

    # Se --full, apaga CSV antigo para começar do zero
    if args.full:
        path_all = os.path.join(output_dir, file_all)
        if os.path.exists(path_all):
            os.remove(path_all)
            print(f"[OK] CSV anterior removido: {path_all}")
        # Também remove CSVs individuais de cada modelo
        for mn in config.MODEL_NAMES:
            slug_csv = os.path.join(output_dir, f"{model_name_to_slug(mn)}.csv")
            if os.path.exists(slug_csv):
                os.remove(slug_csv)

    save_header_if_needed(os.path.join(output_dir, file_all), header)

    print(f"Usando GPU? {torch.cuda.is_available()}")
    print(f"Dispositivo: {config.device}")
    if config.device == "cpu":
        print(">>> Rodando em CPU (mais lento). Para usar GPU: pip install torch --index-url ...")
    elif torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Carregar datasets
    DATASETS = {}
    DATASET_REAL_SIZE = {}
    for name, loader in DATASETS_INFO:
        try:
            DATASETS[name] = loader()
            DATASET_REAL_SIZE[name] = len(DATASETS[name])
            print(f"[OK] {name}: {DATASET_REAL_SIZE[name]} amostras")
        except Exception as e:
            print(f"[ERRO] Erro ao carregar o dataset '{name}': {e}")

    batch_size_map = get_batch_sizes()
    # Por padrão: só roda modelo×dataset que ainda não foram salvos no CSV
    done_pairs = set() if args.full else load_done_pairs(output_dir, file_all)
    if done_pairs:
        print(f"[OK] {len(done_pairs)} combinações já salvas serão puladas (rodando só o que falta). Use --full para rodar tudo.")

    # Avaliação: um modelo por vez
    for model_name in config.MODEL_NAMES:
        try:
            model, tokenizer = load_single_model(model_name)
            print(f"[OK] Modelo carregado: {model_name}")
        except Exception as e:
            print(f"[ERRO] Erro ao carregar modelo '{model_name}': {e}")
            continue

        for dataset_name in DATASETS:
            batch_size = batch_size_map.get((dataset_name, model_name), 1)
            evaluate_model_on_dataset(
                dataset_name,
                DATASETS[dataset_name],
                DATASET_REAL_SIZE.get(dataset_name, ""),
                model_name,
                model,
                tokenizer,
                batch_size,
                output_dir,
                file_all,
                header,
                batch_size_map,
                done_pairs,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print(f"[OK] Modelo descarregado: {model_name}")

    print(f"\nResultados em: {output_dir}")
    print(f"  - Todos: {file_all}")
    print(f"  - Por modelo: <modelo_slug>.csv")


if __name__ == "__main__":
    main()

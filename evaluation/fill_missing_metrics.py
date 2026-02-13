# -*- coding: utf-8 -*-
"""
Preenche apenas as métricas faltantes (COMET, BERTScore F1) para linhas que já têm
modelo×dataset avaliados (re-executa a tradução para obter hipóteses e calcula só essas métricas).

Uso: python -m evaluation.fill_missing_metrics [caminho_csv] [--output saida.csv]
     Se --output não for passado, atualiza o arquivo in-place.
"""
import argparse
import csv
import gc
import os
import sys

import torch

# Garantir que a raiz do projeto está no path
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from evaluation import config
from evaluation.config import get_batch_sizes
from evaluation.datasets import DATASETS_INFO
from evaluation.metrics import compute_metrics
from evaluation.models_loader import load_single_model


def _ensure_metric_columns(fieldnames):
    """Garante que COMET e BERTScore F1 existem no cabeçalho (antes de Erro)."""
    out = list(fieldnames)
    if "COMET" not in out:
        if "Erro" in out:
            i = out.index("Erro")
            out.insert(i, "COMET")
        else:
            out.append("COMET")
    if "BERTScore F1" not in out:
        if "Erro" in out:
            i = out.index("Erro")
            out.insert(i, "BERTScore F1")
        else:
            out.append("BERTScore F1")
    return out


def _save_rows(out_path, fieldnames, rows):
    """Grava as linhas no CSV."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            if "COMET" not in row:
                row["COMET"] = ""
            if "BERTScore F1" not in row:
                row["BERTScore F1"] = ""
            w.writerow(row)


def _needs_fill(row, has_comet_col, has_bs_col):
    """True se a linha tem modelo×dataset válidos e falta COMET ou BERTScore."""
    if (row.get("Erro") or "").strip():
        return False
    model = (row.get("Modelo") or "").strip()
    dataset = (row.get("Dataset") or "").strip()
    if not model or not dataset:
        return False
    if has_comet_col and (row.get("COMET") or "").strip():
        comet_ok = True
    else:
        comet_ok = False
    if has_bs_col and (row.get("BERTScore F1") or "").strip():
        bs_ok = True
    else:
        bs_ok = False
    return not (comet_ok and bs_ok)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Preenche COMET e BERTScore F1 para linhas já avaliadas (re-executa tradução e calcula só essas métricas)."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "evaluation_results",
            "translation_metrics_all.csv"
        ),
        help="CSV com resultados (ex.: evaluation_results/translation_metrics_all.csv)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Arquivo de saída. Se omitido, atualiza o CSV in-place.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        metavar="N",
        help="Limita datasets a N amostras (evita OOM em modelos pesados, ex: danhsf).",
    )
    args = parser.parse_args(argv)

    csv_path = os.path.abspath(args.csv_path)
    out_path = os.path.abspath(args.output) if args.output else csv_path

    if not os.path.exists(csv_path):
        print(f"Arquivo nao encontrado: {csv_path}")
        return 1

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames_orig = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        print("CSV vazio.")
        return 0

    fieldnames = _ensure_metric_columns(fieldnames_orig)
    has_comet = "COMET" in fieldnames_orig
    has_bs = "BERTScore F1" in fieldnames_orig

    # Índices das linhas que precisam preencher e (modelo, dataset) -> [índices]
    need_fill = []  # list of (row_index, model, dataset)
    for i, row in enumerate(rows):
        if _needs_fill(row, has_comet, has_bs):
            need_fill.append((i, (row.get("Modelo") or "").strip(), (row.get("Dataset") or "").strip()))

    if not need_fill:
        print("Nenhuma linha precisa de COMET/BERTScore F1 (todas ja preenchidas ou sem dados).")
        return 0

    # Agrupar por modelo para carregar cada modelo uma vez
    by_model = {}  # model_name -> [(row_index, dataset_name), ...]
    for idx, model, dataset in need_fill:
        if model not in by_model:
            by_model[model] = []
        by_model[model].append((idx, dataset))

    print(f"Preenchendo metricas em {len(need_fill)} linha(s), {len(by_model)} modelo(s).")
    print(f"Dispositivo: {config.device}")

    # Carregar datasets
    DATASETS = {}
    for name, loader in DATASETS_INFO:
        try:
            DATASETS[name] = loader()
            print(f"[OK] Dataset: {name} ({len(DATASETS[name])} amostras)")
        except Exception as e:
            print(f"[ERRO] Dataset '{name}': {e}")

    batch_sizes = get_batch_sizes()

    for model_name, pairs in by_model.items():
        # Ordem do CSV: processar datasets na sequência em que aparecem
        datasets_to_run = [ds for _, ds in sorted(pairs, key=lambda x: x[0])]
        if not all(ds in DATASETS for ds in datasets_to_run):
            print(f"[AVISO] Modelo {model_name}: algum dataset nao carregado, pulando.")
            continue
        try:
            model, tokenizer = load_single_model(model_name)
            print(f"[OK] Modelo carregado: {model_name}")
        except Exception as e:
            print(f"[ERRO] Modelo '{model_name}': {e}")
            continue

        for dataset_name in datasets_to_run:
            data = DATASETS[dataset_name]
            if args.max_samples is not None:
                n = min(args.max_samples, len(data))
                data = data.select(range(n))
                print(f"  [INFO] {dataset_name}: limitado a {len(data)} amostras (--max-samples)")
            batch_size = batch_sizes.get((dataset_name, model_name), 1)
            try:
                bleu, chrf, comet_score, bertscore_f1, elapsed, erro_msg = compute_metrics(
                    data, model, tokenizer, batch_size,
                    dataset_name, model_name, config.device
                )
            except Exception as e:
                print(f"  [ERRO] {model_name} x {dataset_name}: {e}")
                for idx, ds in pairs:
                    if ds == dataset_name:
                        rows[idx]["COMET"] = ""
                        rows[idx]["BERTScore F1"] = ""
                        rows[idx]["Erro"] = str(e)
                _save_rows(out_path, fieldnames, rows)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            comet_str = f"{comet_score:.4f}" if comet_score is not None else ""
            bs_str = f"{bertscore_f1:.4f}" if bertscore_f1 is not None else ""
            for idx, ds in pairs:
                if ds == dataset_name:
                    rows[idx]["COMET"] = comet_str
                    rows[idx]["BERTScore F1"] = bs_str
            print(f"  [OK] {model_name} x {dataset_name}: COMET={comet_str or '-'} BERTScore F1={bs_str or '-'}")
            _save_rows(out_path, fieldnames, rows)
            print(f"      -> CSV atualizado")

        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[OK] Modelo descarregado: {model_name}")

    _save_rows(out_path, fieldnames, rows)
    print(f"Resultado salvo em: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

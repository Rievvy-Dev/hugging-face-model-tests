# -*- coding: utf-8 -*-
"""
Analisa o CSV de avaliação e sugere os 2 melhores modelos para fine-tuning.

Ranking baseado em score composto:
  score = 0.30*BLEU_norm + 0.25*chrF_norm + 0.25*COMET_norm + 0.20*BERTScore_norm

Também mostra ranking individual por cada métrica.

Uso: python choose_best_model.py [caminho_csv]
Padrão: evaluation_results/translation_metrics_all.csv
"""
import csv
import os
import sys
from collections import defaultdict

# Pesos do score composto (soma = 1.0)
W_BLEU = 0.30
W_CHRF = 0.25
W_COMET = 0.25
W_BERTSCORE = 0.20


def _parse_time(s):
    """Extrai segundos de 'Tempo Total' ex: '468.46s'."""
    if not s or not isinstance(s, str):
        return None
    s = s.strip().rstrip("s")
    try:
        return float(s)
    except ValueError:
        return None


def _normalize(values):
    """Min-max normalization para [0, 1]. Retorna lista normalizada."""
    if not values:
        return []
    mn, mx = min(values), max(values)
    if mx == mn:
        return [1.0] * len(values)
    return [(v - mn) / (mx - mn) for v in values]


def main():
    default_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "evaluation_results",
        "translation_metrics_all.csv"
    )
    path = sys.argv[1] if len(sys.argv) > 1 else default_path
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except FileNotFoundError:
        print(f"Arquivo nao encontrado: {path}")
        print("  Rode primeiro: python models-test.py")
        return
    if not rows:
        print("CSV vazio.")
        return

    col_bleu = "BLEU"
    col_chrf = "chr-F"
    col_comet = "COMET"
    col_bertscore = "BERTScore F1"
    col_model = "Modelo"
    col_time = "Tempo Total"
    col_dataset = "Dataset"

    keys_0 = list(rows[0].keys())
    has_comet = col_comet in keys_0
    has_bertscore = col_bertscore in keys_0

    # Agrupar por modelo
    by_model = defaultdict(list)
    for r in rows:
        model = (r.get(col_model) or "").strip()
        if not model or (r.get("Erro") or "").strip():
            continue
        try:
            bleu = float(r.get(col_bleu, 0) or 0)
            chrf = float(r.get(col_chrf, 0) or 0)
            comet_raw = (r.get(col_comet) or "").strip()
            comet = float(comet_raw) if comet_raw else None
            bs_raw = (r.get(col_bertscore) or "").strip()
            bertscore = float(bs_raw) if bs_raw else None
            time_sec = _parse_time(r.get(col_time, ""))
        except ValueError:
            continue
        by_model[model].append({
            "bleu": bleu, "chrf": chrf, "comet": comet,
            "bertscore": bertscore, "time_sec": time_sec,
            "dataset": r.get(col_dataset, ""),
        })

    if not by_model:
        print("Nenhuma linha valida com BLEU/chr-F encontrada.")
        return

    # Calcular médias por modelo
    models = []
    for model, vals in by_model.items():
        n = len(vals)
        avg_bleu = sum(v["bleu"] for v in vals) / n
        avg_chrf = sum(v["chrf"] for v in vals) / n
        comet_vals = [v["comet"] for v in vals if v["comet"] is not None]
        avg_comet = sum(comet_vals) / len(comet_vals) if comet_vals else None
        bs_vals = [v["bertscore"] for v in vals if v["bertscore"] is not None]
        avg_bertscore = sum(bs_vals) / len(bs_vals) if bs_vals else None
        time_vals = [v["time_sec"] for v in vals if v["time_sec"] is not None]
        total_time = sum(time_vals) if time_vals else None
        avg_time = sum(time_vals) / len(time_vals) if time_vals else None
        datasets = sorted(set(v["dataset"] for v in vals))
        models.append({
            "name": model, "n": n,
            "bleu": avg_bleu, "chrf": avg_chrf,
            "comet": avg_comet, "bertscore": avg_bertscore,
            "avg_time": avg_time, "total_time": total_time,
            "datasets": datasets,
        })

    # ---------- Score composto (normalizado) ----------
    bleu_vals = [m["bleu"] for m in models]
    chrf_vals = [m["chrf"] for m in models]
    comet_vals = [m["comet"] if m["comet"] is not None else 0.0 for m in models]
    bs_vals = [m["bertscore"] if m["bertscore"] is not None else 0.0 for m in models]

    bleu_norm = _normalize(bleu_vals)
    chrf_norm = _normalize(chrf_vals)
    comet_norm = _normalize(comet_vals)
    bs_norm = _normalize(bs_vals)

    any_comet = any(m["comet"] is not None for m in models)
    any_bs = any(m["bertscore"] is not None for m in models)

    for i, m in enumerate(models):
        w_bleu, w_chrf, w_comet, w_bs = W_BLEU, W_CHRF, W_COMET, W_BERTSCORE
        # Se faltam métricas, redistribui pesos
        if not any_comet and not any_bs:
            w_bleu, w_chrf, w_comet, w_bs = 0.55, 0.45, 0.0, 0.0
        elif not any_comet:
            w_bleu, w_chrf, w_comet, w_bs = 0.35, 0.30, 0.0, 0.35
        elif not any_bs:
            w_bleu, w_chrf, w_comet, w_bs = 0.35, 0.30, 0.35, 0.0

        m["score"] = (
            w_bleu * bleu_norm[i] +
            w_chrf * chrf_norm[i] +
            w_comet * comet_norm[i] +
            w_bs * bs_norm[i]
        )

    models.sort(key=lambda m: m["score"], reverse=True)

    # ---------- Exibição ----------
    sep = "=" * 90
    print(sep)
    print("  RANKING GERAL - Score composto (BLEU + chr-F + COMET + BERTScore)")
    print(sep)
    for i, m in enumerate(models, 1):
        marker = " *" if i <= 2 else ""
        print(f"\n  {i}. {m['name']}{marker}")
        line = f"     Score: {m['score']:.4f}  |  BLEU: {m['bleu']:.2f}  |  chr-F: {m['chrf']:.2f}"
        if m["comet"] is not None:
            line += f"  |  COMET: {m['comet']:.4f}"
        if m["bertscore"] is not None:
            line += f"  |  BERTScore: {m['bertscore']:.4f}"
        print(line)
        time_line = ""
        if m["avg_time"] is not None:
            time_line = f"     Tempo medio: {m['avg_time']:.1f}s/dataset  |  Total: {m['total_time']:.1f}s"
        print(f"     Datasets: {', '.join(m['datasets'])} ({m['n']} avaliacoes){('  |  ' + time_line.strip()) if time_line else ''}")

    # ---------- Rankings individuais ----------
    metrics_to_rank = [
        ("BLEU", "bleu", True),
        ("chr-F", "chrf", True),
    ]
    if any_comet:
        metrics_to_rank.append(("COMET", "comet", True))
    if any_bs:
        metrics_to_rank.append(("BERTScore F1", "bertscore", True))

    print(f"\n{sep}")
    print("  RANKINGS POR METRICA INDIVIDUAL")
    print(sep)
    for label, key, higher_better in metrics_to_rank:
        ranked = sorted(
            [m for m in models if m.get(key) is not None],
            key=lambda m: m[key], reverse=higher_better
        )
        print(f"\n  {label} (media):")
        for i, m in enumerate(ranked, 1):
            val = m[key]
            fmt = f"{val:.4f}" if key in ("comet", "bertscore") else f"{val:.2f}"
            marker = " <--" if i <= 2 else ""
            print(f"    {i}. {m['name']:55s}  {fmt}{marker}")

    # ---------- Top 2 ----------
    top2 = models[:2]
    print(f"\n{sep}")
    print("  TOP 2 MODELOS RECOMENDADOS PARA FINE-TUNING")
    print(sep)
    for i, m in enumerate(top2, 1):
        print(f"\n  {i}o lugar: {m['name']}")
        line = f"     Score: {m['score']:.4f}  |  BLEU: {m['bleu']:.2f}  |  chr-F: {m['chrf']:.2f}"
        if m["comet"] is not None:
            line += f"  |  COMET: {m['comet']:.4f}"
        if m["bertscore"] is not None:
            line += f"  |  BERTScore: {m['bertscore']:.4f}"
        print(line)

    # Sugestões de fine-tuning
    print(f"\n{sep}")
    print("  SUGESTOES DE FINE-TUNING")
    print(sep)
    for i, m in enumerate(top2, 1):
        name = m["name"]
        print(f"\n  {i}o lugar: {name}")
        if "helsinki" in name.lower() or "opus-mt" in name.lower():
            print("     python helsinki-trainning.py --epochs 3 --output_dir ./models/opus-mt-en-pt-finetuned")
            print("     Com corpus cientifico: python helsinki-trainning.py --extra_csv ./corpus_cientifico.csv --epochs 3")
        elif "mbart" in name.lower():
            print("     Fine-tuning de mBART requer script customizado (Seq2SeqTrainer do HuggingFace).")
            print("     Exemplo: python finetune_mbart.py --model_name {name} --epochs 3")
        elif "t5" in name.lower():
            print("     Fine-tuning de T5: usar Seq2SeqTrainer com prefix 'translate English to Portuguese: '.")
        else:
            print(f"     Sem script dedicado. Use Seq2SeqTrainer do HuggingFace para fine-tuning de {name}.")
    print()


if __name__ == "__main__":
    main()

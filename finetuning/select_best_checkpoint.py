# -*- coding: utf-8 -*-
"""Script para selecionar o melhor checkpoint baseado em métricas."""
import csv
import glob
from pathlib import Path

def select_best_checkpoint(metric="bleu", higher_is_better=True):
    """
    Seleciona o melhor checkpoint baseado em uma métrica.
    
    Args:
        metric: Nome da métrica para comparar (bleu, chrf, comet, bertscore)
        higher_is_better: Se True, maior é melhor; se False, menor é melhor
    
    Returns:
        dict: Informações do melhor checkpoint
    """
    # Buscar todos os CSVs de checkpoint
    pattern = "scielo_after_finetuning_epoch_*.csv"
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        print("❌ Nenhum arquivo de checkpoint encontrado!")
        return None
    
    print(f"\n📊 Analisando {len(csv_files)} checkpoint(s)...\n")
    
    best_checkpoint = None
    best_value = float('-inf') if higher_is_better else float('inf')
    all_results = []
    
    for csv_file in sorted(csv_files):
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    value = float(row[metric])
                    all_results.append({
                        "file": csv_file,
                        "model": row["model"],
                        "checkpoint": row["checkpoint"],
                        "epoch": row["epoch"],
                        "bleu": float(row["bleu"]),
                        "chrf": float(row["chrf"]),
                        "comet": float(row["comet"]),
                        "bertscore": float(row["bertscore"]),
                    })
                    
                    is_better = (value > best_value) if higher_is_better else (value < best_value)
                    
                    if is_better:
                        best_value = value
                        best_checkpoint = all_results[-1]
                
                except (ValueError, KeyError) as e:
                    print(f"⚠️  Erro ao ler {csv_file}: {e}")
                    continue
    
    if not best_checkpoint:
        print("❌ Nenhum checkpoint válido encontrado!")
        return None
    
    # Mostrar todos os resultados
    print(f"{'Época':<10} {'Checkpoint':<20} {'BLEU':<8} {'chr-F':<8} {'COMET':<8} {'BERTScore':<10}")
    print("-" * 80)
    
    for result in all_results:
        marker = "👑" if result == best_checkpoint else "  "
        print(f"{marker} {result['epoch']:<8} {result['checkpoint']:<20} "
              f"{result['bleu']:<8.2f} {result['chrf']:<8.2f} "
              f"{result['comet']:<8.4f} {result['bertscore']:<10.4f}")
    
    print("\n" + "=" * 80)
    print(f"🏆 MELHOR CHECKPOINT (baseado em {metric.upper()}):")
    print("=" * 80)
    print(f"  📁 Arquivo: {best_checkpoint['file']}")
    print(f"  🔢 Época: {best_checkpoint['epoch']}")
    print(f"  📦 Checkpoint: {best_checkpoint['checkpoint']}")
    print(f"  📊 Métricas:")
    print(f"     - BLEU: {best_checkpoint['bleu']:.2f}")
    print(f"     - chr-F: {best_checkpoint['chrf']:.2f}")
    print(f"     - COMET: {best_checkpoint['comet']:.4f}")
    print(f"     - BERTScore: {best_checkpoint['bertscore']:.4f}")
    print()
    
    return best_checkpoint


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Selecionar melhor checkpoint")
    parser.add_argument(
        "--metric",
        type=str,
        default="bleu",
        choices=["bleu", "chrf", "comet", "bertscore"],
        help="Métrica para selecionar o melhor (default: bleu)"
    )
    
    args = parser.parse_args()
    select_best_checkpoint(metric=args.metric)

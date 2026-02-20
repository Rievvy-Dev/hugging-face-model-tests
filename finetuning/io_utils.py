# -*- coding: utf-8 -*-
"""Utilidades de leitura/escrita de arquivos."""
import os
import csv
from . import config


def read_metrics_csv(metrics_file):
    """
    Lê arquivo CSV de métricas.
    
    Args:
        metrics_file: Path ao arquivo CSV
    
    Returns:
        dict: {model_name: {metric_name: value}}
    """
    if not os.path.exists(metrics_file):
        print(f"⚠️  Arquivo não encontrado: {metrics_file}")
        return {}
    
    results = {}
    with open(metrics_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_name = row.get("model", row.get("modelo", "")).strip()
            if model_name:
                results[model_name] = {
                    "bleu": float(row.get("bleu", 0)),
                    "chrf": float(row.get("chrf", 0)),
                    "comet": float(row.get("comet", 0)) if row.get("comet") else None,
                    "bertscore": float(row.get("bertscore", 0)) if row.get("bertscore") else None,
                }
    
    return results


def write_metrics_csv(metrics_data, output_file, fieldnames=None):
    """
    Escreve métricas em arquivo CSV.
    
    Args:
        metrics_data: list de dicts com chaves 'model', 'bleu', 'chrf', etc.
        output_file: Path ao arquivo de saída
        fieldnames: Nomes das colunas (default: auto-detect from data)
    """
    if not metrics_data:
        print(f"⚠️  Nenhum dado para escrever em {output_file}")
        return
    
    if fieldnames is None:
        fieldnames = list(metrics_data[0].keys())
    
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_data)
    
    print(f"✅ Métricas salvas: {output_file}")


def read_checkpoint_status(checkpoint_dir):
    """
    Lê status de checkpoint anterior.
    
    Args:
        checkpoint_dir: Diretório de checkpoint
    
    Returns:
        dict: {model_name: {completed: bool, path: str}}
    """
    status = {}
    
    if os.path.exists(checkpoint_dir):
        for model_dir in os.listdir(checkpoint_dir):
            model_path = os.path.join(checkpoint_dir, model_dir)
            if os.path.isdir(model_path):
                # Verificar se tem checkpoint
                checkpoint_files = [f for f in os.listdir(model_path) if f.startswith("checkpoint-")]
                
                status[model_dir] = {
                    "exists": True,
                    "path": model_path,
                    "checkpoint_count": len(checkpoint_files),
                    "latest_checkpoint": max(checkpoint_files) if checkpoint_files else None,
                }
    
    return status


def checkpoint_exists(model_name, checkpoint_dir):
    """
    Verifica se checkpoint existe para modelo.
    
    Args:
        model_name: Nome do modelo (ex: 'helsinki', 'm2m100')
        checkpoint_dir: Diretório de checkpoints
    
    Returns:
        str or None: Path ao checkpoint se existir, None caso contrário
    """
    status = read_checkpoint_status(checkpoint_dir)
    return status.get(model_name, {}).get("latest_checkpoint")

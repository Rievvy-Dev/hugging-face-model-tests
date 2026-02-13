# -*- coding: utf-8 -*-
"""
Launcher da avaliação: delega para evaluation.run.
Resultados em evaluation_results/translation_metrics_all.csv e por modelo em evaluation_results/<modelo>.csv.

Métricas: BLEU, chr-F, COMET, BERTScore F1 (via unbabel-comet e bert-score).

Uso: python models-test.py [--resume] [--full]
     --resume: pula combinações já salvas (padrão)
     --full:   roda todas as combinações do zero
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation.run import main

if __name__ == "__main__":
    sys.exit(main() or 0)

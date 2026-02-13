# -*- coding: utf-8 -*-
"""
Launcher: preenche COMET e BERTScore F1 nas linhas que já têm modelo×dataset avaliados.
Re-executa a tradução para obter hipóteses e calcula só as métricas faltantes.

Uso: python fill_missing_metrics.py [caminho_csv] [--output saida.csv]
     Ex.: python fill_missing_metrics.py evaluation_results/translation_metrics_all.csv
     Se --output não for passado, atualiza o arquivo in-place.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation.fill_missing_metrics import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

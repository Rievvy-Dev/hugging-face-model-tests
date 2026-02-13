# -*- coding: utf-8 -*-
"""Escrita de CSV e sanitização."""
import csv
import os


def save_header_if_needed(filename, header):
    if not os.path.exists(filename):
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        with open(filename, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)


def sanitize_csv_cell(x, max_len=1500):
    """Evita OSError 22 no Windows: remove quebras de linha e limita tamanho."""
    s = "" if x is None else str(x).strip()
    s = s.replace("\r", " ").replace("\n", " ")
    return s[:max_len] if len(s) > max_len else s


def print_table(data):
    print("\n" + "-" * 150)
    for metric, value in zip(data["Metric"], data["Valor"]):
        print(f"| {metric.ljust(40)} | {str(value).ljust(100)} |")
    print("-" * 150)

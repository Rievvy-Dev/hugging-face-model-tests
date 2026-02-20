# -*- coding: utf-8 -*-
"""
Prepara o dataset Scielo para fine-tuning.
Extrai abstracts paralelos EN-PT e salva em CSV.

O dataset Scielo está disponível em: https://huggingface.co/datasets/scielo
Contém abstracts científicos de diferentes áreas (Biological, Health Sciences, etc).

Uso: python prepare_scielo_dataset.py [--output abstracts_scielo.csv] [--lang en-pt]
"""
import os
import csv
import argparse
import sys
from datasets import load_dataset
from tqdm import tqdm


def prepare_scielo_dataset(output_file="abstracts_scielo.csv", lang_pair="en-pt", max_samples=None):
    """
    Carrega Scielo dataset do HF e extrai abstracts paralelos EN-PT.
    
    Args:
        output_file: caminho do CSV de saída
        lang_pair: par de idiomas (ex: 'en-pt', 'en-es', etc)
        max_samples: limita número de samples (None = sem limite)
    """
    print(f"[1/4] Carregando Scielo dataset ({lang_pair}) do HuggingFace...")
    
    try:
        # Tenta carregar o dataset Scielo
        # O dataset pode estar em diferentes formatos; tentaremos o padrão
        dataset = load_dataset(
            "scielo",
            lang_pair,
            split="train"
        )
    except Exception as e:
        print(f"[AVISO] Erro ao carregar Scielo dataset do HF: {e}")
        print("        Tentando formato alternativo...")
        try:
            # Tenta apenas com o código da língua
            dataset = load_dataset(
                "scielo",
                f"{lang_pair.split('-')[0]}-{lang_pair.split('-')[1]}",
                split="train"
            )
        except Exception as e2:
            print(f"[ERRO] Não foi possível carregar Scielo dataset. Detalhes: {e2}")
            print("       Verifique se o dataset está disponível em https://huggingface.co/datasets/scielo")
            sys.exit(1)
    
    print(f"    ✅ Dataset carregado: {len(dataset)} exemplos")
    print(f"    Colunas: {dataset.column_names}")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"    Limitado a {len(dataset)} exemplos")
    
    # =========================================
    # Extração de abstracts paralelos
    # =========================================
    print(f"\n[2/4] Extraindo abstracts paralelos...")
    
    abstracts = []
    skipped_no_abstract = 0
    skipped_empty = 0
    
    for example in tqdm(dataset, desc="Processando", ncols=100):
        # Formatos possíveis dependem da estrutura do dataset Scielo
        # O Scielo usa: {"translation": {"en": "...", "pt": "..."}}
        
        en_text = None
        pt_text = None
        
        # Tentar diferentes nomes de colunas (em ordem de probabilidade)
        
        # Caso 1: translation com dicionário (NOVO - Scielo official format)
        if "translation" in example and isinstance(example["translation"], dict):
            trans = example["translation"]
            en_text = trans.get("en") or trans.get("source") or trans.get("English")
            pt_text = trans.get("pt") or trans.get("target") or trans.get("Portuguese")
        
        # Caso 2: source_texts / target_texts
        elif "source_texts" in example and "target_texts" in example:
            en_text = example.get("source_texts")
            pt_text = example.get("target_texts")
        
        # Caso 3: en / pt diretos
        elif "en" in example and "pt" in example:
            en_text = example.get("en")
            pt_text = example.get("pt")
        
        # Caso 4: texts como dicionário
        elif "texts" in example and isinstance(example["texts"], dict):
            texts = example["texts"]
            en_text = texts.get("en") or texts.get("source")
            pt_text = texts.get("pt") or texts.get("target")
        
        # Caso 5: tentar adivinhar (última tentativa)
        else:
            cols = [k for k in example.keys() if k not in ["id", "metadata", "document_id", "translation"]]
            if len(cols) >= 2:
                en_text = example.get(cols[0])
                pt_text = example.get(cols[1])
        
        # Normaliza texto
        if en_text:
            en_text = str(en_text).strip() if en_text else None
        if pt_text:
            pt_text = str(pt_text).strip() if pt_text else None
        
        # Filtra
        if not en_text or not pt_text:
            skipped_no_abstract += 1
            continue
        
        if len(en_text) < 20 or len(pt_text) < 20:  # mínimo razoável
            skipped_empty += 1
            continue
        
        abstracts.append({"abstract_en": en_text, "abstract_pt": pt_text})
    
    print(f"    ✅ Extraído: {len(abstracts)} pares de abstracts")
    print(f"    Pulados (sem abstract): {skipped_no_abstract}")
    print(f"    Pulados (muito curtos): {skipped_empty}")
    
    if not abstracts:
        print("[ERRO] Nenhum abstract foi extraído. Verifique a estrutura do dataset.")
        sys.exit(1)
    
    # =========================================
    # Salvar em CSV
    # =========================================
    print(f"\n[3/4] Salvando em CSV: {output_file}")
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["abstract_en", "abstract_pt"])
        writer.writeheader()
        for row in abstracts:
            writer.writerow(row)
    
    print(f"    ✅ Salvos {len(abstracts)} pares em: {output_file}")
    
    # =========================================
    # Estatísticas
    # =========================================
    print(f"\n[4/4] Estatísticas:")
    
    en_lengths = [len(a["abstract_en"].split()) for a in abstracts]
    pt_lengths = [len(a["abstract_pt"].split()) for a in abstracts]
    
    print(f"    EN - Palavras por abstract: min={min(en_lengths)}, med={sum(en_lengths)//len(en_lengths)}, max={max(en_lengths)}")
    print(f"    PT - Palavras por abstract: min={min(pt_lengths)}, med={sum(pt_lengths)//len(pt_lengths)}, max={max(pt_lengths)}")
    print(f"    Total de palavras EN: {sum(en_lengths)}")
    print(f"    Total de palavras PT: {sum(pt_lengths)}")
    
    print(f"\n✅ Dataset Scielo preparado com sucesso!")
    print(f"   Arquivo: {output_file}")
    print(f"   Exemplos: {len(abstracts)}")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara Scielo dataset para fine-tuning")
    parser.add_argument("--output", type=str, default="abstracts_scielo.csv", help="Arquivo CSV de saída")
    parser.add_argument("--lang", type=str, default="en-pt", help="Par de idiomas (ex: en-pt, en-es, en-fr)")
    parser.add_argument("--max_samples", type=int, default=None, help="Limite de samples (para teste rápido)")
    
    args = parser.parse_args()
    
    prepare_scielo_dataset(output_file=args.output, lang_pair=args.lang, max_samples=args.max_samples)

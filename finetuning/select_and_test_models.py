# -*- coding: utf-8 -*-
"""
Script para testar/avaliar modelos de tradução selecionados (sem fine-tuning).

Uso:
  # Preparar dados e testar modelos base
  python finetuning/select_and_test_models.py
  
  # Testar modelos base com dados já preparados
  python finetuning/select_and_test_models.py --skip_prepare
  
  # Testar modelo específico (helsinki ou m2m100)
  python finetuning/select_and_test_models.py --skip_prepare --model helsinki
  
  # Testar modelos fine-tuned
  python finetuning/select_and_test_models.py --test_finetuned
  
  # Testar modelo específico fine-tuned
  python finetuning/select_and_test_models.py --test_finetuned --model m2m100
  
  # Testar ambos (base e fine-tuned) e comparar
  python finetuning/select_and_test_models.py --test_both
"""

import os
import sys
import argparse

# Adicionar parent directory ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finetuning import config, data_utils, evaluate, compare


def parse_args():
    """Parse argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Script para avaliar modelos de tradução selecionados"
    )

    # Dataset
    parser.add_argument(
        "--abstracts",
        type=str,
        default=config.SCIELO_ABSTRACTS_FILE,
        help="Arquivo CSV com abstracts (default: abstracts_scielo.csv)"
    )
    parser.add_argument(
        "--train_samples",
        type=int,
        default=config.DEFAULT_TRAIN_SAMPLES,
        help=f"Exemplos para TREINO (default: {config.DEFAULT_TRAIN_SAMPLES})"
    )
    parser.add_argument(
        "--val_samples",
        type=int,
        default=config.DEFAULT_VAL_SAMPLES,
        help=f"Exemplos para VALIDACAO (default: {config.DEFAULT_VAL_SAMPLES})"
    )
    parser.add_argument(
        "--test_samples",
        type=int,
        default=config.DEFAULT_TEST_SAMPLES,
        help=f"Exemplos para TESTE (default: {config.DEFAULT_TEST_SAMPLES})"
    )

    # Modelo
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=list(config.MODELS.keys()),
        help="Modelo específico para testar (default: todos)"
    )

    # Controle de execução
    parser.add_argument(
        "--skip_prepare", 
        action="store_true", 
        help="Pula preparo de dados (usa arquivos existentes)"
    )
    parser.add_argument(
        "--test_finetuned", 
        action="store_true", 
        help="Testa apenas modelos fine-tuned (após treinamento)"
    )
    parser.add_argument(
        "--test_both", 
        action="store_true", 
        help="Testa modelos base E fine-tuned, e compara resultados"
    )

    return parser.parse_args()


def main():
    """Executa avaliação de modelos."""
    args = parse_args()

    print("\n" + "="*80)
    print("  TESTE DE MODELOS DE TRADUÇÃO EN→PT")
    print("="*80 + "\n")

    # Verificar arquivo de dados
    if not args.skip_prepare and not os.path.exists(args.abstracts):
        print(f"[ERRO] Arquivo não encontrado: {args.abstracts}")
        print("Execute primeiro: python prepare_scielo_dataset.py")
        sys.exit(1)

    # Preparar dados (splits train/val/test)
    if not args.skip_prepare:
        print("📊 Preparando dados de teste...\n")
        train_csv, val_csv, test_csv = datasets.prepare_evaluation_csv(
            abstracts_file=args.abstracts,
            train_csv=config.SCIELO_TRAIN_CSV,
            val_csv=config.SCIELO_VAL_CSV,
            test_csv=config.SCIELO_TEST_CSV,
            train_samples=args.train_samples,
            val_samples=args.val_samples,
            test_samples=args.test_samples,
        )
    else:
        test_csv = config.SCIELO_TEST_CSV
        print(f"✅ Usando dados existentes: {test_csv}\n")

    # Determinar quais modelos testar
    model_name = args.model
    if model_name:
        print(f"🎯 Testando modelo específico: {model_name}\n")
    
    if args.test_both:
        print("🔍 Testando MODELOS BASE e MODELOS FINE-TUNED\n")
        evaluate.evaluate_before(test_csv=test_csv, model_name=model_name)
        evaluate.evaluate_after(test_csv=test_csv, model_name=model_name)
        
        # Comparar resultados
        print("\n" + "="*80)
        print("  COMPARAÇÃO: Base vs Fine-tuned")
        print("="*80 + "\n")
        
        compare.compare_and_report(
            before_file=config.BEFORE_METRICS_FILE,
            after_file=config.AFTER_METRICS_FILE,
            output_file=config.COMPARISON_REPORT,
        )
        
    elif args.test_finetuned:
        print("🔍 Testando MODELOS FINE-TUNED\n")
        evaluate.evaluate_after(test_csv=test_csv, model_name=model_name)
        
    else:
        print("🔍 Testando MODELOS BASE (originais)\n")
        evaluate.evaluate_before(test_csv=test_csv, model_name=model_name)

    print("\n" + "="*80)
    print("  ✅ AVALIAÇÃO CONCLUÍDA!")
    print("="*80)
    
    # Mostrar arquivos gerados
    if args.test_both:
        print(f"\n📁 Resultados salvos:")
        print(f"   - Base: {config.BEFORE_METRICS_FILE}")
        print(f"   - Fine-tuned: {config.AFTER_METRICS_FILE}")
        print(f"   - Comparação: {config.COMPARISON_REPORT}")
    elif args.test_finetuned:
        print(f"\n📁 Resultados salvos: {config.AFTER_METRICS_FILE}")
    else:
        print(f"\n📁 Resultados salvos: {config.BEFORE_METRICS_FILE}")
    
    print()


if __name__ == "__main__":
    main()

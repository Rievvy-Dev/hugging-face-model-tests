# -*- coding: utf-8 -*-
"""
Script para fine-tuning de modelos de tradução selecionados.

Uso:
  # Fine-tuning completo (prepara dados + treina todos os modelos)
  python finetuning/finetune_selected_models.py
  
  # Fine-tuning com dados já preparados
  python finetuning/finetune_selected_models.py --skip_prepare
  
  # Fine-tuning de modelo específico
  python finetuning/finetune_selected_models.py --model helsinki
  
  # Fine-tuning com parâmetros customizados
  python finetuning/finetune_selected_models.py --epochs 10 --batch_size 16 --lr 5e-5
  
  # Retomar fine-tuning interrompido
  python finetuning/finetune_selected_models.py --resume_from ./models/finetuned-scielo/helsinki/checkpoint-1000
"""

import os
import sys
import argparse

# Adicionar parent directory ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finetuning import config, data_utils, trainer


def parse_args():
    """Parse argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Script para fine-tuning de modelos de tradução selecionados"
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
        help="Modelo específico para fine-tuning (default: todos)"
    )

    # Hiperparâmetros
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.DEFAULT_EPOCHS,
        help=f"Épocas (default: {config.DEFAULT_EPOCHS})"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config.DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {config.DEFAULT_BATCH_SIZE})"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=config.DEFAULT_LR,
        help=f"Learning rate (default: {config.DEFAULT_LR})"
    )

    # Controle de execução
    parser.add_argument(
        "--skip_prepare", 
        action="store_true", 
        help="Pula preparo de dados (usa arquivos existentes)"
    )
    parser.add_argument(
        "--resume_from", 
        type=str, 
        default=None, 
        help="Path do checkpoint para retomar treinamento"
    )

    return parser.parse_args()


def main():
    """Executa fine-tuning de modelos selecionados."""
    args = parse_args()

    print("\n" + "="*80)
    print("  FINE-TUNING DE MODELOS DE TRADUÇÃO EN→PT")
    print("="*80 + "\n")

    # Verificar arquivo de dados
    if not args.skip_prepare and not os.path.exists(args.abstracts):
        print(f"[ERRO] Arquivo não encontrado: {args.abstracts}")
        print("Execute primeiro: python prepare_scielo_dataset.py")
        sys.exit(1)

    # Preparar dados (splits train/val/test)
    if not args.skip_prepare:
        print("📊 Preparando dados para fine-tuning...\n")
        train_csv, val_csv, test_csv = data_utils.prepare_evaluation_csv(
            abstracts_file=args.abstracts,
            train_csv=config.SCIELO_TRAIN_CSV,
            val_csv=config.SCIELO_VAL_CSV,
            test_csv=config.SCIELO_TEST_CSV,
            train_samples=args.train_samples,
            val_samples=args.val_samples,
            test_samples=args.test_samples,
        )
    else:
        train_csv = config.SCIELO_TRAIN_CSV
        val_csv = config.SCIELO_VAL_CSV
        print(f"✅ Usando dados existentes:")
        print(f"   - Treino: {train_csv}")
        print(f"   - Validação: {val_csv}\n")

    # Determinar quais modelos treinar
    if args.model:
        model_names = [args.model]
        print(f"🎯 Fine-tuning de 1 modelo: {args.model}\n")
    else:
        model_names = list(config.MODELS.keys())
        print(f"🎯 Fine-tuning de {len(model_names)} modelos: {', '.join(model_names)}\n")

    # Informações de configuração
    print(f"⚙️  Hiperparâmetros:")
    print(f"   - Épocas: {args.epochs}")
    print(f"   - Batch size: {args.batch_size}")
    print(f"   - Learning rate: {args.lr}")
    print(f"   - Device: {config.device}")
    
    if args.resume_from:
        print(f"\n🔄 Retomando treinamento de: {args.resume_from}")
    
    print("\n" + "-"*80 + "\n")

    # Fine-tuning de cada modelo
    failed_models = []
    
    for i, model_name in enumerate(model_names, 1):
        print(f"\n{'='*80}")
        print(f"  MODELO {i}/{len(model_names)}: {model_name}")
        print(f"{'='*80}\n")
        
        output_dir = f"./models/finetuned-scielo/{model_name}"
        
        result = trainer.finetune_model(
            model_name=model_name,
            train_csv=train_csv,
            val_csv=val_csv,
            output_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            resume_from_checkpoint=args.resume_from,
        )

        if not result.get("success"):
            failed_models.append(model_name)
            print(f"\n❌ [ERRO] {result.get('message', 'Falha no fine-tuning')}")
            
            if result.get("checkpoint"):
                print(f"\n💾 Checkpoint salvo. Para retomar:")
                print(f"   python finetuning/finetune_selected_models.py --model {model_name} "
                      f"--resume_from {result['checkpoint']} --skip_prepare\n")
            
            # Perguntar se quer continuar com próximo modelo
            if i < len(model_names):
                response = input("\n⚠️  Continuar com próximo modelo? (s/N): ")
                if response.lower() != 's':
                    print("\n🛑 Processo interrompido pelo usuário.")
                    sys.exit(1)
        else:
            print(f"\n✅ Modelo {model_name} fine-tuned com sucesso!")
            print(f"   📁 Salvo em: {output_dir}\n")

    # Resumo final
    print("\n" + "="*80)
    print("  RESUMO DO FINE-TUNING")
    print("="*80 + "\n")
    
    successful = len(model_names) - len(failed_models)
    
    print(f"✅ Modelos completados: {successful}/{len(model_names)}")
    
    if failed_models:
        print(f"❌ Modelos com falha: {', '.join(failed_models)}")
    
    print(f"\n💡 Próximo passo:")
    print(f"   python finetuning/select_and_test_models.py --test_finetuned")
    print(f"   ou")
    print(f"   python finetuning/select_and_test_models.py --test_both\n")


if __name__ == "__main__":
    main()

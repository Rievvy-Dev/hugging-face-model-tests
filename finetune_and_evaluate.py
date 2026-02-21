# -*- coding: utf-8 -*-
"""
Pipeline completo de fine-tuning e avaliacao com suporte a resume.

Uso:
  python finetune_and_evaluate.py --train_samples 200000 --epochs 5
"""

import os
import sys
import argparse
from finetuning import config, data_utils, trainer, evaluate, compare


def parse_args():
    """Parse argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Pipeline completo de fine-tuning e avaliacao com suporte a resume"
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

    # Fine-tuning
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.DEFAULT_EPOCHS,
        help=f"Epocas (default: {config.DEFAULT_EPOCHS})"
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

    # Controle de execucao
    parser.add_argument("--skip_prepare", action="store_true", help="Pula preparo de dados")
    parser.add_argument("--skip_before", action="store_true", help="Pula avaliacao ANTES")
    parser.add_argument("--skip_finetune", action="store_true", help="Pula fine-tuning")
    parser.add_argument("--skip_after", action="store_true", help="Pula avaliacao DEPOIS")
    parser.add_argument("--resume_from", type=str, default=None, help="Checkpoint para retomar")

    return parser.parse_args()


def main():
    """Executa o pipeline completo."""
    args = parse_args()

    if not args.skip_prepare and not os.path.exists(args.abstracts):
        print(f"[ERRO] Arquivo nao encontrado: {args.abstracts}")
        print("Execute primeiro: python prepare_scielo_dataset.py")
        sys.exit(1)

    train_csv, val_csv, test_csv = data_utils.prepare_evaluation_csv(
        abstracts_file=args.abstracts,
        train_csv=config.SCIELO_TRAIN_CSV,
        val_csv=config.SCIELO_VAL_CSV,
        test_csv=config.SCIELO_TEST_CSV,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        test_samples=args.test_samples,
    )

    if not args.skip_before:
        evaluate.evaluate_before(test_csv=test_csv)

    if not args.skip_finetune:
        for model_name in config.MODELS.keys():
            result = trainer.finetune_model(
                model_name=model_name,
                train_csv=train_csv,
                output_dir=f"./models/finetuned-scielo/{model_name}",
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                resume_from_checkpoint=args.resume_from,
            )

            if not result.get("success"):
                print(f"[ERRO] {result.get('message', 'Falha no fine-tuning')}")
                if result.get("checkpoint"):
                    print(
                        "Para retomar, use: python finetune_and_evaluate.py "
                        f"--resume_from {result['checkpoint']} --skip_before --skip_prepare"
                    )
                sys.exit(1)

    if not args.skip_after:
        evaluate.evaluate_after(test_csv=test_csv)

    compare.compare_and_report(
        before_file=config.BEFORE_METRICS_FILE,
        after_file=config.AFTER_METRICS_FILE,
        output_file=config.COMPARISON_REPORT,
    )


if __name__ == "__main__":
    main()

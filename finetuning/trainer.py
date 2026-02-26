# -*- coding: utf-8 -*-
"""Fine-tuning com suporte a checkpoints e resume."""
import os
import inspect
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from datasets import Dataset
from tqdm import tqdm
from . import config
from .io_utils import checkpoint_exists, read_checkpoint_status


def prepare_dataset_for_training(csv_file, tokenizer, model_name, max_samples=None, max_seq_len=None):
    """
    Prepara dataset para fine-tuning.
    
    Args:
        csv_file: Path ao CSV com abstract_en / abstract_pt
        tokenizer: Tokenizador
        model_name: Nome do modelo (para detectar se precisa lang code)
        max_samples: Limitar número de amostras (None = todas)
    
    Returns:
        Dataset pronto para treinamento
    """
    import csv
    
    # Carregar CSV
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    if max_samples:
        data = data[:max_samples]
    
    print(f"✅ Carregado: {len(data):,} exemplos de {csv_file}")
    
    # Verificar se precisa lang code (M2M100)
    requires_lang = "m2m100" in model_name.lower()
    
    max_seq_len = max_seq_len or config.DEFAULT_MAX_SEQ_LEN

    def process_data(examples):
        """Processa exemplos para HF Dataset."""
        inputs = []
        targets = []
        
        for item in examples:
            src = item.get("abstract_en", "").strip()
            tgt = item.get("abstract_pt", "").strip()
            
            if src and tgt:
                # Adicionar lang code se necessário
                if requires_lang:
                    src = "__pt_BR__" + src
                
                inputs.append(src)
                targets.append(tgt)
        
        # Tokenizar
        model_inputs = tokenizer(
            inputs,
            max_length=max_seq_len,
            truncation=True,
            padding="max_length",
        )
        
        # Tokenizar targets e mascarar PAD para nao contribuir na loss
        labels = tokenizer(
            targets,
            max_length=max_seq_len,
            truncation=True,
            padding="max_length",
        )
        pad_id = tokenizer.pad_token_id
        masked_labels = [
            [(token if token != pad_id else -100) for token in seq]
            for seq in labels["input_ids"]
        ]

        model_inputs["labels"] = masked_labels
        
        return model_inputs
    
    # Criar HF Dataset
    dataset = Dataset.from_dict({
        "abstract_en": [d.get("abstract_en", "") for d in data],
        "abstract_pt": [d.get("abstract_pt", "") for d in data],
    })
    
    # Processar
    dataset = dataset.map(
        lambda x: process_data([{"abstract_en": en, "abstract_pt": pt} 
                                for en, pt in zip(x["abstract_en"], x["abstract_pt"])]),
        batched=True,
        batch_size=32,
        remove_columns=["abstract_en", "abstract_pt"]
    )
    
    return dataset


def finetune_model(model_name, 
                   train_csv=config.SCIELO_TRAIN_CSV,
                   val_csv=config.SCIELO_VAL_CSV,
                   output_dir=None,
                   epochs=config.DEFAULT_EPOCHS,
                   batch_size=config.DEFAULT_BATCH_SIZE,
                   lr=config.DEFAULT_LR,
                   max_seq_len=config.DEFAULT_MAX_SEQ_LEN,
                   grad_accum_steps=config.DEFAULT_GRAD_ACCUM_STEPS,
                   fp16=config.DEFAULT_FP16,
                   max_steps=None,
                   early_stopping_patience=config.DEFAULT_EARLY_STOPPING_PATIENCE,
                   resume_from_checkpoint=None):
    """
    Fine-tuna modelo no dataset Scielo com suporte a resume.
    
    ⭐ IMPORTANTE: Se interrompido, pode retomar com resume_from_checkpoint=<path>
    
    Args:
        model_name: Nome do modelo ('helsinki' ou 'm2m100')
        train_csv: Path ao CSV de treino
        output_dir: Diretório para salvar modelo fine-tuned
        epochs: Número de épocas
        batch_size: Batch size para treino
        lr: Learning rate
        resume_from_checkpoint: Path a checkpoint anterior para retomar
    
    Returns:
        dict: {success: bool, model_path: str, message: str}
    """
    
    if output_dir is None:
        output_dir = f"./models/finetuned-scielo/{model_name}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"  Fine-tuning: {model_name}")
    print(f"{'='*80}\n")
    
    print(f"  📍 Configuração:")
    print(f"     ├─ Treino: {train_csv}")
    print(f"     ├─ Validacao: {val_csv}")
    print(f"     ├─ Epochs: {epochs}")
    print(f"     ├─ Batch size: {batch_size}")
    print(f"     ├─ Learning rate: {lr}")
    print(f"     ├─ Max seq len: {max_seq_len}")
    print(f"     ├─ Grad accum steps: {grad_accum_steps}")
    print(f"     ├─ FP16: {fp16}")
    print(f"     └─ Output: {output_dir}\n")
    
    # Carregar modelo e tokenizador
    model_path = config.MODELS.get(model_name, model_name)
    
    print(f"  📦 Carregando modelo: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    # Gradient checkpointing para reduzir memoria
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print(f"     ✅ Gradient checkpointing ativado")
    
    # Preparar datasets
    print(f"\n  📚 Preparando datasets...")
    train_dataset = prepare_dataset_for_training(
        train_csv,
        tokenizer,
        model_name,
        max_seq_len=max_seq_len,
    )
    eval_dataset = None
    if val_csv and os.path.exists(val_csv):
        eval_dataset = prepare_dataset_for_training(
            val_csv,
            tokenizer,
            model_name,
            max_seq_len=max_seq_len,
        )
    else:
        print("     ⚠️  Validação nao encontrada. Seguindo sem eval_dataset.")
    
    print(f"     Treino: {len(train_dataset)} exemplos")
    print(f"     Validação: {len(eval_dataset) if eval_dataset is not None else 0} exemplos\n")
    
    # Training arguments com suporte a resume e early stopping
    eval_strategy = "epoch" if eval_dataset is not None else "no"
    use_fp16 = fp16 and config.device == "cuda"
    use_early_stopping = eval_dataset is not None and early_stopping_patience > 0
    training_args_kwargs = dict(
        output_dir=output_dir,
        overwrite_output_dir=False,  # Manter checkpoints
        num_train_epochs=epochs if max_steps is None else 1,
        max_steps=max_steps if max_steps is not None else -1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_steps=config.DEFAULT_WARMUP_STEPS,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=2,  # Manter apenas 2 checkpoints
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss" if eval_dataset is not None else None,
        greater_is_better=False,
        gradient_accumulation_steps=grad_accum_steps,
        fp16=use_fp16,
        logging_steps=100,
        predict_with_generate=True,
        optim="adamw_torch",
        seed=config.SEED,
        report_to=[],  # Sem wandb/tensorboard
    )
    init_params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    if "evaluation_strategy" in init_params:
        training_args_kwargs["evaluation_strategy"] = eval_strategy
    else:
        training_args_kwargs["eval_strategy"] = eval_strategy

    training_args = Seq2SeqTrainingArguments(
        **training_args_kwargs
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    # Early stopping callback
    callbacks = []
    if use_early_stopping:
        from transformers import EarlyStoppingCallback
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=config.DEFAULT_EARLY_STOPPING_THRESHOLD,
        ))
        print(f"     ✅ Early stopping ativado (patience={early_stopping_patience})")
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    
    # Treinar com resume
    try:
        print(f"🚀 Iniciando fine-tuning...")
        print(f"   ⏳ Progresso será exibido pelo transformers...\n")
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Salvar modelo final
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print(f"\n✅ Fine-tuning finalizado!")
        print(f"   📁 Modelo salvo: {output_dir}")
        print(f"   📊 Loss final: {train_result.training_loss:.4f}\n")
        
        return {
            "success": True,
            "model_path": output_dir,
            "message": f"Fine-tuning de {model_name} concluído com sucesso"
        }
    
    except KeyboardInterrupt:
        print(f"\n⏸️  Fine-tuning interrompido!")
        print(f"   💾 Checkpoints salvos em: {output_dir}")
        print(f"   🔄 Retomar com: resume_from_checkpoint='{output_dir}/checkpoint-XXX'\n")
        
        # Encontrar último checkpoint
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            latest = sorted(checkpoints)[-1]
            checkpoint_path = os.path.join(output_dir, latest)
            return {
                "success": False,
                "model_path": output_dir,
                "checkpoint": checkpoint_path,
                "message": f"Fine-tuning interrompido. Retomar com: {checkpoint_path}"
            }
    
    except Exception as e:
        print(f"\n❌ Erro durante fine-tuning: {e}")
        return {
            "success": False,
            "model_path": output_dir,
            "message": f"Erro: {str(e)}"
        }

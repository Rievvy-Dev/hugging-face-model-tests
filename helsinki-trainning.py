# helsinki-trainning.py
import os
import math
import shutil
import argparse
import time
import json
import csv
import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from datasets.utils.logging import set_verbosity_error
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)
import evaluate
import numpy as np

set_verbosity_error()  # reduz verbosidade do datasets

# =========================
# Utilidades de Log
# =========================
class EpochTimerCallback(TrainerCallback):
    def __init__(self):
        self.epoch_times = {}   # epoch_idx -> seconds
        self._epoch_start = None
        self._last_epoch = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._epoch_start = time.time()
        self._last_epoch = int(state.epoch) if state.epoch is not None else None

    def on_epoch_end(self, args, state, control, **kwargs):
        if self._epoch_start is not None:
            e = int(state.epoch) if state.epoch is not None else (self._last_epoch or 0)
            self.epoch_times[e] = self.epoch_times.get(e, 0.0) + (time.time() - self._epoch_start)
            self._epoch_start = None


def save_training_log_csv(log_history, csv_path):
    """
    Salva trainer.state.log_history em CSV (sem pandas).
    Acha todas as chaves que aparecem ao longo do histórico e grava colunas fixas.
    """
    # Coletar todas as chaves
    keys = set()
    for row in log_history:
        keys.update(row.keys())
    # Ordenar chaves (metadados por último)
    key_order = sorted([k for k in keys if k not in ["runtime", "train_runtime"]]) + \
                [k for k in ["runtime", "train_runtime"] if k in keys]

    # Escrever CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=key_order)
        writer.writeheader()
        for row in log_history:
            # converter valores não-serializáveis
            safe_row = {}
            for k in key_order:
                v = row.get(k, "")
                if isinstance(v, (list, dict)):
                    v = json.dumps(v, ensure_ascii=False)
                safe_row[k] = v
            writer.writerow(safe_row)


def save_summary_txt(path, info_dict):
    with open(path, "w", encoding="utf-8") as f:
        f.write("=== Treinamento Helsinki EN→PT (MarianMT) ===\n")
        for k, v in info_dict.items():
            f.write(f"{k}: {v}\n")


# =========================
# Datasets
# =========================
def load_two_datasets(max_ted=None, max_tatoeba=None, seed=42):
    print("[1/8] Carregando OPUS100 (en-pt) …")
    ted = load_dataset("opus100", "en-pt", split="train")
    print(f"      OPUS100 colunas: {ted.column_names}")
    if max_ted:
        ted = ted.shuffle(seed=seed).select(range(min(max_ted, len(ted))))
    # garantir coluna translation apenas (paranoia)
    if "translation" not in ted.features:
        ted = ted.map(lambda ex: {"translation": ex["translation"]},
                      remove_columns=[c for c in ted.column_names if c != "translation"])
    print("      Exemplo OPUS100:", ted[0])

    print("[2/8] Carregando Tatoeba (en-pt) …")
    tatoeba = None
    errors = []

    # Tente primeiro a config que o cache do cluster reportou existir
    try:
        tatoeba = load_dataset("tatoeba", "en-pt-lang1=en,lang2=pt", split="train", trust_remote_code=True)
        print("      Tatoeba: usando config 'en-pt-lang1=en,lang2=pt'")
    except Exception as e:
        errors.append(f"en-pt-lang1=en,lang2=pt: {e}")

    # Alternativa comum
    if tatoeba is None:
        try:
            tatoeba = load_dataset("tatoeba", "en-pt", split="train", trust_remote_code=True)
            print("      Tatoeba: usando config 'en-pt'")
        except Exception as e:
            errors.append(f"en-pt: {e}")

    if tatoeba is None:
        raise RuntimeError("Falha ao carregar Tatoeba. Tentativas: " + " | ".join(errors))

    print(f"      Tatoeba colunas: {tatoeba.column_names}")
    if max_tatoeba:
        tatoeba = tatoeba.shuffle(seed=seed).select(range(min(max_tatoeba, len(tatoeba))))
    try:
        print("      Exemplo Tatoeba bruto:", {k: tatoeba[0][k] for k in tatoeba.column_names})
    except Exception:
        pass

    # Normalização robusta do Tatoeba para {"translation":{"en":..., "pt":...}}
    def _to_translation(ex):
        # Caso 1: já é dict translation
        if "translation" in ex and isinstance(ex["translation"], dict):
            tr = ex["translation"]
            en = tr.get("en") or tr.get("source") or tr.get("sentence_en") or tr.get("text_en")
            pt = tr.get("pt") or tr.get("target") or tr.get("sentence_pt") or tr.get("text_pt")
            if en is not None and pt is not None:
                return {"translation": {"en": en, "pt": pt}}
        # Caso 2: pares explícitos
        if "source" in ex and "target" in ex:
            return {"translation": {"en": ex["source"], "pt": ex["target"]}}
        if "en" in ex and "pt" in ex:
            return {"translation": {"en": ex["en"], "pt": ex["pt"]}}
        if "sentence1" in ex and "sentence2" in ex:
            return {"translation": {"en": ex["sentence1"], "pt": ex["sentence2"]}}
        if "text" in ex and "translation_text" in ex:
            return {"translation": {"en": ex["text"], "pt": ex["translation_text"]}}
        # Heurística por prefixo
        en_key = next((k for k in ex.keys() if k.lower().startswith("en")), None)
        pt_key = next((k for k in ex.keys() if k.lower().startswith("pt")), None)
        if en_key and pt_key:
            return {"translation": {"en": ex[en_key], "pt": ex[pt_key]}}
        # Se não deu, marca vazio
        return {"translation": {"en": None, "pt": None}}

    try:
        tatoeba = tatoeba.map(_to_translation, desc="Normalizando Tatoeba", num_proc=max(1, (os.cpu_count() or 1)))
    except Exception as e:
        print(f"      Aviso: multiprocessing falhou ({e}). Refazendo com num_proc=1…")
        tatoeba = tatoeba.map(_to_translation, desc="Normalizando Tatoeba", num_proc=1)

    tatoeba = tatoeba.filter(lambda ex: ex["translation"]["en"] is not None and ex["translation"]["pt"] is not None)
    tatoeba = tatoeba.remove_columns([c for c in tatoeba.column_names if c != "translation"])
    print("      Exemplo Tatoeba normalizado:", tatoeba[0])

    print("[3/8] Concatenando OPUS100 + Tatoeba …")
    full = concatenate_datasets([ted, tatoeba])

    print("[4/8] Criando split treino/teste (90/10)…")
    split = full.train_test_split(test_size=0.1, seed=seed)

    print("[5/8] Tamanhos — train:", len(split["train"]), " | test:", len(split["test"]))
    return split


# =========================
# Tokenização
# =========================
def tokenize_function(batch, tokenizer, max_length=128):
    ens = [ex["en"] for ex in batch["translation"]]
    pts = [ex["pt"] for ex in batch["translation"]]
    model_inputs = tokenizer(ens, max_length=max_length, truncation=True)
    labels = tokenizer(text_target=pts, max_length=max_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def postprocess_text(preds, labels):
    preds = [p.strip() for p in preds]
    labels = [l.strip() for l in labels]
    return preds, labels


# =========================
# Main
# =========================
def main():
    # CUDA check dentro da main (evita prints duplicados)
    if not torch.cuda.is_available():
        raise SystemExit("❌ CUDA não disponível! Este script exige GPU com CUDA.")
    device = torch.device("cuda")
    print(f"✅ Usando dispositivo: {device} ({torch.cuda.get_device_name(0)})")
    globals()['device'] = device

    # (Opcional) TF32 para Nvidia Ampere+ (melhora throughput)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./models/opus-mt-en-pt-finetuned")
    parser.add_argument("--zip_path", default="./models/opus-mt-en-pt-finetuned.zip")
    parser.add_argument("--max_ted", type=int, default=None)
    parser.add_argument("--max_tatoeba", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--train_bs", type=int, default=16)
    parser.add_argument("--eval_bs", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=128)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    split = load_two_datasets(max_ted=args.max_ted, max_tatoeba=args.max_tatoeba)

    print("[6/8] Carregando tokenizer/modelo Helsinki…")
    model_name = "Helsinki-NLP/opus-mt-tc-big-en-pt"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)

    print("▶️ Tokenizando datasets…")
    tokenized_train = split["train"].map(
        lambda x: tokenize_function(x, tokenizer, max_length=args.max_len),
        batched=True, remove_columns=split["train"].column_names
    )
    tokenized_eval = split["test"].map(
        lambda x: tokenize_function(x, tokenizer, max_length=args.max_len),
        batched=True, remove_columns=split["test"].column_names
    )
    print(f"   ✅ Tokenização concluída | train: {len(tokenized_train)} | eval: {len(tokenized_eval)}")

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    bleu = evaluate.load("bleu")
    chrf = evaluate.load("chrf")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        bleu_score = bleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])["bleu"] * 100
        chrf_score = chrf.compute(predictions=decoded_preds, references=decoded_labels)["score"]
        return {"bleu": bleu_score, "chrf": chrf_score}

    print("[7/8] Configurando treinamento…")
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        generation_max_length=args.max_len,
        fp16=True,  # usa mixed precision na GPU
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
    )

    timer_cb = EpochTimerCallback()
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[timer_cb],
    )

    print("🚀 Iniciando treino…")
    t0 = time.time()
    train_output = trainer.train()
    total_time = time.time() - t0
    print(f"⏱️ Tempo total de treino: {total_time:.2f}s")

    print("💾 Salvando modelo/tokenizer…")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # -------------------------
    # Salvar LOG do Treino (CSV) e SUMMARY (TXT)
    # -------------------------
    log_csv_path = os.path.join(args.output_dir, "training_log.csv")
    print(f"📝 Salvando log de treino em CSV: {log_csv_path}")
    save_training_log_csv(trainer.state.log_history, log_csv_path)

    summary_path = os.path.join(args.output_dir, "summary.txt")
    best_metrics = train_output.metrics if hasattr(train_output, "metrics") else {}
    # Tenta pegar última avaliação do log_history
    last_eval = None
    for row in reversed(trainer.state.log_history):
        if "eval_bleu" in row or "eval_loss" in row:
            last_eval = row
            break

    summary_info = {
        "Modelo Base": model_name,
        "Output Dir": args.output_dir,
        "Epochs": args.epochs,
        "Learning Rate": args.lr,
        "Batch Size (train/eval)": f"{args.train_bs}/{args.eval_bs}",
        "Max Length": args.max_len,
        "Tamanho Train (amostrado)": len(tokenized_train),
        "Tamanho Eval (amostrado)": len(tokenized_eval),
        "Tempo Total (s)": f"{total_time:.2f}",
        "Tempos por Época (s)": json.dumps(timer_cb.epoch_times),
        "Melhor Métrica (treinador)": json.dumps(best_metrics, ensure_ascii=False),
        "Última Avaliação": json.dumps(last_eval, ensure_ascii=False),
    }
    print(f"🧾 Salvando resumo: {summary_path}")
    save_summary_txt(summary_path, summary_info)

    # -------------------------
    # Compactar pasta do modelo em ZIP
    # -------------------------
    print(f"📦 Compactando pasta para ZIP único: {args.zip_path}")
    if os.path.exists(args.zip_path):
        os.remove(args.zip_path)
    shutil.make_archive(args.zip_path.replace(".zip",""), "zip", args.output_dir)

    print("✅ Fine-tuning concluído!")
    print(f"   Pasta do modelo: {args.output_dir}")
    print(f"   ZIP único:       {args.zip_path}")
    print(f"   Log CSV:         {log_csv_path}")
    print(f"   Summary:         {summary_path}")


if __name__ == "__main__":
    main()

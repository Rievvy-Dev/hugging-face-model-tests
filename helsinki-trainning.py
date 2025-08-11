# helsinki-trainning.py
import os
import math
import shutil
import argparse
import time
import json
import csv
import sys
import torch
from datasets import load_dataset, Dataset, concatenate_datasets, DownloadConfig
from datasets.utils.logging import set_verbosity_error
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint
import evaluate
import numpy as np

set_verbosity_error()  # reduz verbosidade do datasets


# =========================
# Callbacks / Utils
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


class EpochCsvLoggerCallback(TrainerCallback):
    """
    Salva um CSV por época com as métricas de avaliação/treino reportadas pelo Trainer.
    Arquivos: epoch_XX_metrics.csv e epoch_XX_train_steps.csv em output_dir.
    """
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        try:
            epoch = int(metrics.get("epoch", state.epoch or 0))
        except Exception:
            epoch = int(state.epoch or 0)
        path = os.path.join(args.output_dir, f"epoch_{epoch:02d}_metrics.csv")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            for k, v in metrics.items():
                if isinstance(v, (list, dict)):
                    v = json.dumps(v, ensure_ascii=False)
                w.writerow([k, v])

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch or 0)
        last_rows = [r for r in state.log_history if isinstance(r, dict) and int(r.get("epoch", -1)) == epoch]
        if not last_rows:
            return
        path = os.path.join(args.output_dir, f"epoch_{epoch:02d}_train_steps.csv")
        keys = sorted({k for row in last_rows for k in row.keys()})
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in last_rows:
                safe_row = {}
                for k in keys:
                    v = row.get(k, "")
                    if isinstance(v, (list, dict)):
                        v = json.dumps(v, ensure_ascii=False)
                    safe_row[k] = v
                w.writerow(safe_row)


def save_training_log_csv(log_history, csv_path):
    keys = set()
    for row in log_history:
        if isinstance(row, dict):
            keys.update(row.keys())
    key_order = sorted([k for k in keys if k not in ["runtime", "train_runtime"]]) + \
                [k for k in ["runtime", "train_runtime"] if k in keys]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=key_order)
        writer.writeheader()
        for row in log_history:
            if not isinstance(row, dict):
                continue
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
    ted = load_dataset(
        "opus100", "en-pt", split="train",
        download_config=DownloadConfig(local_files_only=False)
    )
    print(f"      OPUS100 colunas: {ted.column_names}")
    if max_ted:
        ted = ted.shuffle(seed=seed).select(range(min(max_ted, len(ted))))
    if "translation" not in ted.features:
        ted = ted.map(lambda ex: {"translation": ex["translation"]},
                      remove_columns=[c for c in ted.column_names if c != "translation"])
    print("      Exemplo OPUS100:", ted[0])

    print("[2/8] Carregando Tatoeba (en-pt) …")
    # Forma correta: dataset 'tatoeba' com lang1/lang2
    tatoeba = load_dataset(
        "tatoeba",
        split="train",
        lang1="en",
        lang2="pt",
        trust_remote_code=True,
        download_config=DownloadConfig(local_files_only=False)
    )
    print(f"      Tatoeba colunas: {tatoeba.column_names}")
    if max_tatoeba:
        tatoeba = tatoeba.shuffle(seed=seed).select(range(min(max_tatoeba, len(tatoeba))))
    try:
        print("      Exemplo Tatoeba bruto:", {k: tatoeba[0][k] for k in tatoeba.column_names})
    except Exception:
        pass

    # Normalização robusta para {"translation":{"en":..., "pt":...}}
    def _to_translation(ex):
        if "translation" in ex and isinstance(ex["translation"], dict):
            tr = ex["translation"]
            en = tr.get("en") or tr.get("source") or tr.get("sentence_en") or tr.get("text_en")
            pt = tr.get("pt") or tr.get("target") or tr.get("sentence_pt") or tr.get("text_pt")
            if en is not None and pt is not None:
                return {"translation": {"en": en, "pt": pt}}
        if "source" in ex and "target" in ex:
            return {"translation": {"en": ex["source"], "pt": ex["target"]}}
        if "en" in ex and "pt" in ex:
            return {"translation": {"en": ex["en"], "pt": ex["pt"]}}
        if "sentence1" in ex and "sentence2" in ex:
            return {"translation": {"en": ex["sentence1"], "pt": ex["sentence2"]}}
        if "text" in ex and "translation_text" in ex:
            return {"translation": {"en": ex["text"], "pt": ex["translation_text"]}}
        en_key = next((k for k in ex.keys() if k.lower().startswith("en")), None)
        pt_key = next((k for k in ex.keys() if k.lower().startswith("pt")), None)
        if en_key and pt_key:
            return {"translation": {"en": ex[en_key], "pt": ex[pt_key]}}
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
    if not torch.cuda.is_available():
        raise SystemExit("❌ CUDA não disponível! Este script exige GPU com CUDA.")
    device = torch.device("cuda")
    print(f"✅ Usando dispositivo: {device} ({torch.cuda.get_device_name(0)})")
    globals()['device'] = device

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

    # Tempo/controle
    parser.add_argument("--time_hours", type=float, default=10.0, help="Orçamento máximo de tempo (horas)")
    parser.add_argument("--safety", type=float, default=0.85, help="Margem de segurança (0-1). 0.85 = usa 85% do tempo alvo")
    parser.add_argument("--calib_steps", type=int, default=120, help="Passos para calibração de s/it")
    parser.add_argument("--calib_samples", type=int, default=4096, help="Exemplos para calibração")
    parser.add_argument("--max_eval", type=int, default=5000, help="Limite do conjunto de avaliação")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Quantos checkpoints manter")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # -------------------------
    # Carregar datasets brutos
    # -------------------------
    split = load_two_datasets(max_ted=args.max_ted, max_tatoeba=args.max_tatoeba)

    # Limita eval para acelerar avaliação por época
    if args.max_eval and len(split["test"]) > args.max_eval:
        split["test"] = split["test"].shuffle(seed=42).select(range(args.max_eval))
        print(f"🔎 Eval limitado para {len(split['test'])} exemplos.")

    # -------------------------
    # Tokenizer (antes da calibração)
    # -------------------------
    print("[6/8] Carregando tokenizer…")
    model_name = "Helsinki-NLP/opus-mt-tc-big-en-pt"
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    # -------------------------
    # Calibração rápida (usa um modelo separado, não altera o de treino)
    # -------------------------
    print("⏱️ Calibrando tempo por passo…")
    req_calib = max(args.calib_steps * args.train_bs, min(args.calib_samples, len(split["train"])))
    req_calib = min(req_calib, len(split["train"]))
    calib_ds = split["train"].shuffle(seed=123).select(range(req_calib))
    tokenized_calib = calib_ds.map(
        lambda x: tokenize_function(x, tokenizer, max_length=args.max_len),
        batched=True, remove_columns=calib_ds.column_names
    )

    calib_model = MarianMTModel.from_pretrained(model_name).to(device)
    calib_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=calib_model)

    steps_target = max(1, min(args.calib_steps, math.floor(len(tokenized_calib) / args.train_bs)))

    calib_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(args.output_dir, "calib_tmp"),
        per_device_train_batch_size=args.train_bs,
        learning_rate=args.lr,
        num_train_epochs=1,
        max_steps=steps_target,          # controla o # de passos na calibração
        logging_strategy="steps",
        logging_steps=10,
        evaluation_strategy="no",
        save_strategy="no",
        dataloader_num_workers=0,
        fp16=True,
        report_to="none",
        disable_tqdm=False,
        overwrite_output_dir=True,
    )

    calib_trainer = Seq2SeqTrainer(
        model=calib_model,
        args=calib_args,
        train_dataset=tokenized_calib,
        tokenizer=tokenizer,
        data_collator=calib_collator,
    )

    torch.cuda.synchronize(); t0 = time.time()
    calib_trainer.train()
    torch.cuda.synchronize(); t1 = time.time()
    elapsed = t1 - t0
    s_per_step = elapsed / max(1, steps_target)
    print(f"   ✅ Calibração: {steps_target} passos em {elapsed:.2f}s → {s_per_step:.3f} s/it")

    # limpa calibração da GPU
    del calib_trainer, calib_model
    torch.cuda.empty_cache()

    # -------------------------
    # Dimensionamento do dataset para caber no tempo
    # -------------------------
    budget_seconds = args.time_hours * 3600 * max(0.0, min(args.safety, 1.0))
    examples_per_epoch = int((budget_seconds / (s_per_step * max(1, args.epochs))) * args.train_bs)
    examples_per_epoch = max(args.train_bs * 32, examples_per_epoch)               # mínimo razoável
    examples_per_epoch = min(examples_per_epoch, len(split["train"]))              # não excede

    print(f"📏 Orçamento: {args.time_hours}h × margem {args.safety:.2f} → {budget_seconds:.0f}s")
    print(f"📐 Estimado: {s_per_step:.3f}s/it | batch={args.train_bs} | épocas={args.epochs} → "
          f"{examples_per_epoch} exemplos/época (~{examples_per_epoch/args.train_bs:.0f} steps/época)")

    # Reamostra o train para o tamanho-alvo
    split["train"] = split["train"].shuffle(seed=42).select(range(examples_per_epoch))

    # -------------------------
    # Tokenização final (train/eval)
    # -------------------------
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

    # -------------------------
    # Carrega o modelo REAL para treino + collator
    # -------------------------
    print("[7/8] Configurando treinamento…")
    model = MarianMTModel.from_pretrained(model_name).to(device)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Métricas (sem walrus)
    bleu = evaluate.load("bleu")
    chrf = evaluate.load("chrf")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]
        bleu_score = bleu.compute(
            predictions=decoded_preds,
            references=[[l] for l in decoded_labels]
        )["bleu"] * 100
        chrf_score = chrf.compute(
            predictions=decoded_preds,
            references=decoded_labels
        )["score"]
        return {"bleu": bleu_score, "chrf": chrf_score}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",              # salva checkpoint a cada época
        logging_strategy="steps",
        logging_steps=100,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        weight_decay=0.01,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.epochs,
        predict_with_generate=True,        # necessário para BLEU/chrF
        generation_max_length=args.max_len,
        fp16=True,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        overwrite_output_dir=True,         # evita erro se pasta já existir
    )

    timer_cb = EpochTimerCallback()
    csv_cb   = EpochCsvLoggerCallback()
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[timer_cb, csv_cb],
    )

    # ---------- Resume só se existir checkpoint ----------
    last_ckpt = None
    if os.path.isdir(args.output_dir):
        try:
            last_ckpt = get_last_checkpoint(args.output_dir)
        except Exception:
            last_ckpt = None
    print(f"↩️  Resume: {'sim, '+last_ckpt if last_ckpt else 'não (sem checkpoint)'}")

    print("🚀 Iniciando treino…")
    t0 = time.time()
    train_output = trainer.train(resume_from_checkpoint=last_ckpt)  # None => treino do zero
    total_time = time.time() - t0
    print(f"⏱️ Tempo total de treino: {total_time:.2f}s")

    print("💾 Salvando modelo/tokenizer…")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # -------------------------
    # LOG e SUMMARY
    # -------------------------
    log_csv_path = os.path.join(args.output_dir, "training_log.csv")
    print(f"📝 Salvando log de treino em CSV: {log_csv_path}")
    save_training_log_csv(trainer.state.log_history, log_csv_path)

    summary_path = os.path.join(args.output_dir, "summary.txt")
    last_eval = None
    for row in reversed(trainer.state.log_history):
        if isinstance(row, dict) and ("eval_bleu" in row or "eval_loss" in row):
            last_eval = row
            break

    est_steps_per_epoch = math.ceil(len(tokenized_train) / args.train_bs)
    est_total_steps = est_steps_per_epoch * args.epochs
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
        "s/it (calibrado)": f"{s_per_step:.4f}",
        "Passos/época (estimado)": est_steps_per_epoch,
        "Passos totais (estimado)": est_total_steps,
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

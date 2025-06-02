import torch
import psutil
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
import evaluate
from tqdm import tqdm
import time
import csv
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando GPU? {torch.cuda.is_available()}")
print(f"Dispositivo: {device}")

dataset = load_dataset("tatoeba", lang1="en", lang2="pt", trust_remote_code=True)["train"]
dataset = dataset.shuffle(seed=42).select(range(1000))

bleu_metric = evaluate.load("bleu")
chrf_metric = evaluate.load("chrf")

model_names = [
    "Helsinki-NLP/opus-mt-tc-big-en-pt",
    "Narrativa/mbart-large-50-finetuned-opus-en-pt-translation",
    "unicamp-dl/translation-en-pt-t5",
    "VanessaSchenkel/unicamp-finetuned-en-to-pt-dataset-ted"
]

output_file = "resultados_traducao.csv"

if not os.path.exists(output_file):
    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Modelo",
            "Tempo Total",
            "Tempo por Sentença (s)",
            "Palavras por Segundo",
            "Uso de Memória (RAM)",
            "Memória GPU Alocada (MB)",
            "Memória GPU Reservada (MB)",
            "Total de Sentenças",
            "Total de Palavras",
            "BLEU",
            "chr-F"
        ])

def compute_metrics(dataset, model, tokenizer, num_samples=1000, batch_size=32):
    model.eval()
    references, hypotheses = [], []

    for i in tqdm(range(0, min(num_samples, len(dataset)), batch_size), desc="Calculando métricas"):
        batch = dataset[i:i+batch_size]
        inputs_text = [item['en'] for item in batch['translation']]
        targets_text = [item['pt'] for item in batch['translation']]

        inputs = tokenizer(inputs_text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)

        with torch.no_grad():
            if device == "cuda":
                with torch.amp.autocast(device_type='cuda'):
                    translated = model.generate(**inputs)
            else:
                translated = model.generate(**inputs)

        decoded_translations = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

        references.extend([[ref] for ref in targets_text])
        hypotheses.extend(decoded_translations)

    bleu_score = bleu_metric.compute(predictions=hypotheses, references=references)["bleu"] * 100
    chrf_score = chrf_metric.compute(predictions=hypotheses, references=references)["score"]

    return bleu_score, chrf_score

def print_table(data):
    print("\n" + "-" * 100)
    for metric, value in zip(data["Metric"], data["Valor"]):
        print(f"| {metric.ljust(45)} | {str(value).ljust(45)} |")
    print("-" * 100)

for model_name in model_names:
    print(f"\nAvaliando o modelo: {model_name}")
    process = psutil.Process()
    memory_before = process.memory_info().rss

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    start_time = time.time()
    bleu_score, chrf_score = compute_metrics(dataset, model, tokenizer)
    end_time = time.time()

    elapsed_time = end_time - start_time
    num_sentences = len(dataset)
    num_words = sum(len(ex["translation"]["pt"].split()) for ex in dataset)
    memory_after = process.memory_info().rss

    gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
    gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 ** 2) if torch.cuda.is_available() else 0

    tempo_por_sentenca = elapsed_time / num_sentences
    palavras_por_segundo = num_words / elapsed_time

    data = {
        "Metric": [
            "Nome do Modelo",
            "Tempo Total",
            "Tempo por Sentença (s)",
            "Palavras por Segundo",
            "Uso de Memória (RAM)",
            "Memória GPU Alocada (MB)",
            "Memória GPU Reservada (MB)",
            "Total de Sentenças",
            "Total de Palavras",
            "BLEU Score",
            "chr-F Score",
        ],
        "Valor": [
            model_name,
            f"{elapsed_time:.2f}s",
            f"{tempo_por_sentenca:.4f}",
            f"{palavras_por_segundo:.2f}",
            f"{(memory_after - memory_before) / (1024 * 1024):.2f} MB",
            f"{gpu_memory_allocated:.2f}",
            f"{gpu_memory_reserved:.2f}",
            num_sentences,
            num_words,
            f"{bleu_score:.2f}",
            f"{chrf_score:.5f}",
        ]
    }

    print_table(data)

    with open(output_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            model_name,
            f"{elapsed_time:.2f}",
            f"{tempo_por_sentenca:.4f}",
            f"{palavras_por_segundo:.2f}",
            f"{(memory_after - memory_before) / (1024 * 1024):.2f}",
            f"{gpu_memory_allocated:.2f}",
            f"{gpu_memory_reserved:.2f}",
            num_sentences,
            num_words,
            f"{bleu_score:.2f}",
            f"{chrf_score:.5f}",
        ])

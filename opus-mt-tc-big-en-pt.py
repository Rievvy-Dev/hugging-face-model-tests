import torch
import psutil
from transformers import MarianMTModel, MarianTokenizer
from datasets import load_dataset
import evaluate
from tqdm import tqdm
import time
from pynvml import *

nvmlInit()

process = psutil.Process()
memory_before = process.memory_info().rss

device = "cuda" if torch.cuda.is_available() else "cpu"

gpu_info = "Nenhuma GPU detectada"
gpu_memory_allocated = 0
gpu_memory_reserved = 0
gpu_usage = 0
if torch.cuda.is_available():
    gpu_info = torch.cuda.get_device_name(0)
    handle = nvmlDeviceGetHandleByIndex(0)
    memory_info = nvmlDeviceGetMemoryInfo(handle)
    gpu_memory_allocated = memory_info.used / (1024 ** 2)
    gpu_memory_reserved = memory_info.total / (1024 ** 2)
    gpu_usage = (gpu_memory_allocated / gpu_memory_reserved) * 100

print(f"Usando: {device}")
print(f"GPU: {gpu_info}")

start_time = time.time()

dataset = load_dataset("tatoeba", lang1="en", lang2="pt", trust_remote_code=True)["train"].shuffle(seed=42).select(range(1000))

tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-pt")
model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-pt")

model = model.to(device)

start_time = time.time()

scaler = torch.cuda.amp.GradScaler()

bleu_metric = evaluate.load("bleu")
chrf_metric = evaluate.load("chrf")

batch_size = 32

def compute_metrics(dataset, model, tokenizer, num_samples=1000):
    model.eval()
    references, hypotheses = [], []

    for i in tqdm(range(0, min(num_samples, len(dataset)), batch_size), desc="Calculando métricas"):

        batch = dataset[i:i+batch_size]  
        en_texts = [ex['en'] for ex in batch]  
        pt_texts = [ex['pt'] for ex in batch]  

        inputs = tokenizer(en_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                translated = model.generate(**inputs)

        decoded_translations = [tokenizer.decode(t[0], skip_special_tokens=True) for t in translated]

        references.extend([[pt_text] for pt_text in pt_texts])
        hypotheses.extend(decoded_translations)

    bleu_score = bleu_metric.compute(predictions=hypotheses, references=references)["bleu"] * 100
    chrf_score = chrf_metric.compute(predictions=hypotheses, references=references)["score"]

    return bleu_score, chrf_score

bleu_score, chrf_score = compute_metrics(dataset, model, tokenizer)

num_sentences = len(dataset)
num_words = sum(len(ex["translation"]["pt"].split()) for ex in dataset)

end_time = time.time()
elapsed_time = end_time - start_time

process = psutil.Process()
memory_after = process.memory_info().rss

data = {
    "Metric": [
        "Tempo de Execução",
        "Uso de Memória",
        "Sentenças",
        "Palavras",
        "BLEU Score",
        "chr-F Score",
        "Uso de GPU (Memória Alocada)",
        "Uso de GPU (Memória Reservada)",
        "Uso de GPU (%)",
        "Batch Size"
    ],
    "Valor": [
        f"{elapsed_time:.2f}s",
        f"{(memory_after - memory_before) / (1024 * 1024):.2f} MB",
        num_sentences,
        num_words,
        f"{bleu_score:.2f}",
        f"{chrf_score:.5f}",
        f"{gpu_memory_allocated:.2f} MB",
        f"{gpu_memory_reserved:.2f} MB",
        f"{gpu_usage:.2f}%",
        batch_size
    ]
}

def print_table(data):
    print("\n" + "-" * 45)
    for metric, value in zip(data["Metric"], data["Valor"]):
        print(f"| {metric.ljust(30)} | {str(value).ljust(10)} |")
    print("-" * 45)

print_table(data)

nvmlShutdown()
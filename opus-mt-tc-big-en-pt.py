import torch
import psutil
from transformers import MarianMTModel, MarianTokenizer
from datasets import load_dataset
import evaluate
from tqdm import tqdm
import time

process = psutil.Process()
memory_before = process.memory_info().rss

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando GPU? {torch.cuda.is_available()}")
print(f"Usando: {device}")

dataset = load_dataset("tatoeba", lang1="en", lang2="pt", trust_remote_code=True)["train"]
dataset = dataset.shuffle(seed=42).select(range(1000))

tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-pt")
model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-pt")

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

model = model.to(device)

start_time = time.time()

bleu_metric = evaluate.load("bleu")
chrf_metric = evaluate.load("chrf")

def compute_metrics(dataset, model, tokenizer, num_samples=1000, batch_size=128):
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

bleu_score, chrf_score = compute_metrics(dataset, model, tokenizer)

end_time = time.time()
elapsed_time = end_time - start_time

num_sentences = len(dataset)
num_words = sum(len(ex["translation"]["pt"].split()) for ex in dataset)

memory_after = process.memory_info().rss

gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 ** 2) if torch.cuda.is_available() else 0

data = {
    "Metric": [
        "Tempo de Execução",
        "Uso de Memória (RAM)",
        "Memória GPU Alocada",
        "Memória GPU Reservada",
        "Sentenças",
        "Palavras",
        "BLEU Score",
        "chr-F Score",
    ],
    "Valor": [
        f"{elapsed_time:.2f}s",
        f"{(memory_after - memory_before) / (1024 * 1024):.2f} MB",
        f"{gpu_memory_allocated:.2f} MB",
        f"{gpu_memory_reserved:.2f} MB",
        num_sentences,
        num_words,
        f"{bleu_score:.2f}",
        f"{chrf_score:.5f}",
    ]
}

def print_table(data):
    print("\n" + "-" * 60)
    for metric, value in zip(data["Metric"], data["Valor"]):
        print(f"| {metric.ljust(35)} | {str(value).ljust(15)} |")
    print("-" * 60)

print_table(data)
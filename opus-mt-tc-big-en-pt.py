import torch
from transformers import MarianMTModel, MarianTokenizer
from datasets import load_dataset
import evaluate
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

dataset = load_dataset("tatoeba", lang1="en", lang2="pt", trust_remote_code=True)["train"].shuffle(seed=42).select(range(1000))

tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-pt")
model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-pt")

# Se houver mais de uma GPU, usa o DataParallel
if torch.cuda.device_count() > 1:
    print("Usando múltiplas GPUs!")
    model = torch.nn.DataParallel(model)

model = model.to(device)

bleu_metric = evaluate.load("bleu")
chrf_metric = evaluate.load("chrf")

def compute_metrics(dataset, model, tokenizer, num_samples=1000):
    model.eval()
    references, hypotheses = [], []
    
    for i in tqdm(range(min(num_samples, len(dataset))), desc="Calculando métricas"):
        en_text = dataset[i]["translation"]["en"]
        pt_text = dataset[i]["translation"]["pt"]
        
        inputs = tokenizer(en_text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        
        with torch.no_grad():
            translated = model.generate(**inputs)
        
        decoded_translation = tokenizer.decode(translated[0], skip_special_tokens=True)
        
        references.append([pt_text])
        hypotheses.append(decoded_translation)
    
    bleu_score = bleu_metric.compute(predictions=hypotheses, references=references)["bleu"] * 100
    chrf_score = chrf_metric.compute(predictions=hypotheses, references=references)["score"]
    
    print(f"\n🔹 BLEU Score: {bleu_score:.2f}")
    print(f"🔹 chr-F Score: {chrf_score:.5f}")

compute_metrics(dataset, model, tokenizer)

num_sentences = len(dataset)
num_words = sum(len(ex["translation"]["pt"].split()) for ex in dataset)

print(f"\n🔹 Sentences: {num_sentences}")
print(f"🔹 Words: {num_words}")
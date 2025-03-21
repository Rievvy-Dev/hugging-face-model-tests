# **Translation Evaluation with BLEU and chr-F**

This project evaluates the **Helsinki-NLP/opus-mt-tc-big-en-pt** translation model using **BLEU** and **chr-F** metrics. The evaluation is based on the **Tatoeba** dataset, specifically for **English-to-Portuguese** translations.

## **🚀 Features**
- ✅ Uses **Hugging Face Transformers** to load and run the MarianMT translation model.
- ✅ Automatically detects **GPU (CUDA)** for faster inference.
- ✅ Loads **1,000 shuffled sentence pairs** from the **Tatoeba** dataset.
- ✅ Evaluates the model using **BLEU and chr-F** scores.
- ✅ Displays the total number of **sentences and words** processed.

## **📌 How It Works**
1. **Loads the dataset** from Hugging Face and selects 1,000 sentence pairs.
2. **Loads the MarianMT model** and tokenizer for English-to-Portuguese translation.
3. **Generates translations** and compares them with the reference translations.
4. **Computes BLEU and chr-F scores** to measure translation quality.
5. **Prints evaluation results** and dataset statistics.

## **🛠 Installation**
Make sure you have Python and the required dependencies installed:

```bash
pip install torch transformers datasets evaluate tqdm
```

## **▶️ Usage**
Run the script:

```bash
python model.py
```

## **📊 Example Output**
```
🔹 BLEU Score: 38.75  
🔹 chr-F Score: 0.56342  
🔹 Sentences: 1000  
🔹 Words: 14,532  
```

## **📜 License**
This project is open-source and available under the MIT License.

# Automatic-Summarization-of-Arabic-Texts-using-Transformers
 Automatic summarization of Arabic texts using Transformer-based models (mT5, AraBART, MBART). This project applies deep learning and NLP techniques to generate concise and meaningful summaries of long Arabic documents.



# 🕌 Automatic Summarization of Arabic Texts using Transformers

##  Overview
This project addresses **automatic abstractive summarization of Arabic texts** using Transformer-based models.  
The goal is to generate **concise, coherent summaries** of long Arabic news articles, leveraging the **XL-Sum dataset** and fine-tuning the **AraBART** model.  

The work contributes to the underexplored field of **Arabic NLP**, where resources are scarce compared to English, and demonstrates that modern Transformers can effectively handle Arabic’s morphological and syntactic richness.

---

## 📚 Dataset: XL-Sum (Arabic Subset)
- **Source:** BBC Arabic news articles  
- **Task type:** Abstractive summarization (single-sentence summaries)  
- **Format:** Each entry contains two fields → `text` (article body) & `summary` (gold summary)  
- **Language:** Arabic only (dataset also available in 44 other languages)  

###  Exploratory Data Analysis
- **Article length:** Most between 1,000–3,000 characters, with some outliers > 40,000 characters.  
- **Summary length:** Mostly 150–250 characters (one-sentence abstractive summaries).  
- This makes XL-Sum highly suitable for **short abstractive summarization** tasks.  

---

## 🧹 Preprocessing Pipeline
To improve data quality before training, a custom Arabic preprocessing pipeline was built:

| Step | Description |
|------|-------------|
| **Duplicate removal** | Remove repeated entries |
| **delete_links** | Remove hyperlinks |
| **delete_repeated_characters** | Normalize repeated punctuation/letters |
| **remove_extra_spaces** | Keep only valid Arabic words |
| **replace_letters** | Normalize similar Arabic letters |
| **clean_text** | Remove non-Arabic / non-numeric characters |
| **remove_vowelization** | Strip diacritics |
| **delete_stopwords** | Remove Arabic + English stopwords |
| **stem_text** | Apply ISRI Arabic stemmer |
| **text_prepare** | Master function orchestrating all steps |

---

## ⚙️ Model: AraBART
We fine-tuned **Jezia/AraBART-finetuned-wiki-ar** on XL-Sum Arabic.

### Tokenization
- **Input text:** Truncated/padded to **1000 tokens**  
- **Summaries:** Truncated/padded to **400 tokens**  
- **Pad tokens** replaced with `-100` to ignore during loss computation  
- **Attention masks** used to focus only on relevant tokens  

### Training Configuration
| Hyperparameter        | Value |
|------------------------|-------|
| Pretrained model       | Jezia/AraBART-finetuned-wiki-ar |
| Optimizer             | AdamW |
| Learning rate         | 1e-4 |
| Batch size            | 2 |
| Max input length      | 1000 |
| Max summary length    | 400 |
| Epochs                | 3 |
| Decoding              | Beam Search (5 beams) |
| Repetition penalty    | 1.0 |
| Length penalty        | 0.8 |
| Evaluation metrics    | ROUGE-L, Semantic similarity (SBERT) |
| Hardware              | Kaggle GPU (CUDA) |

Training was conducted with **PyTorch Lightning**, using `trainer.fit()` for efficient optimization.

---

## 📊 Evaluation Results
Two evaluation metrics were used:

- **ROUGE-L** → measures lexical overlap between generated and reference summaries  
- **Semantic Similarity** → cosine similarity of embeddings (Sentence-BERT)  

### Results (AraBART)
- **ROUGE-L:** Most scores between **0.15 and 0.40**  
- **Semantic similarity:** Concentrated between **0.90 and 1.0**  

✅ Interpretation: Even when lexical overlap is moderate, the generated summaries preserve the **semantic meaning** of the original references.

---

## 🚀 Usage
### Install dependencies
```bash
pip install -r requirements.txt

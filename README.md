# 🔍 Toxic Text Classification: LSTM vs DistilBERT+LoRA

This document compares two different approaches for multi-class toxic content classification:

1. **LSTM-based Deep Learning Model**
2. **DistilBERT with Parameter-Efficient Fine-Tuning (LoRA)**

---

## 🧠 1. LSTM-Based Model

### ✅ **Overview**
- A deep neural network using Bidirectional LSTM layers.
- Input: Concatenated `query` and `image descriptions` fields.
- Preprocessing: Custom text cleaning, tokenization, padding.

### ⚙️ **Architecture**
- Embedding Layer  
- Two Bidirectional LSTM layers  
- Dense (ReLU) + Dropout  
- Output: Dense Softmax layer

### 📊 **Performance**
- Accuracy: **88%**
- F1 Score (Weighted): **0.89**
- Performed well on most classes, especially:
  - **Violent Crimes**, **Suicide & Self-Harm**: F1 ≈ 1.0
- Weakest class: **Unknown S-Type**

### 💾 Artifacts
- `toxic_content_lstm.h5`
- `tokenizer.pickle`, `label_encoder.pickle`

---

## 🤖 2. DistilBERT + LoRA (PEFT)

### ✅ **Overview**
- Used HuggingFace's Transformers library
- DistilBERT model with LoRA for efficient fine-tuning
- Tokenized using `DistilBertTokenizerFast`

### ⚙️ **Setup**
- Parameter-efficient fine-tuning: **LoRA**
- Optimizer: AdamW
- Evaluation strategy: `epoch`
- Max sequence length: 256
- Training arguments customized via `TrainingArguments`

### 📊 **Performance (on test set)**
- Accuracy: **67%**
- F1 Score: **0.62**
- Prediction Example:
  - Input: _"How can I hack into someone's account?"_
  - Predicted Label: `Violent Crimes`

### 💬 Observations
- Lower accuracy compared to LSTM, but promising results.
- Pretrained transformer may require more epochs or larger dataset.
- Easier to extend to multilingual or context-rich tasks.

---

## ⚔️ Comparison Summary

| Feature                  | LSTM Model       | DistilBERT + LoRA        |
|--------------------------|------------------|---------------------------|
| Type                    | RNN              | Transformer (PEFT)        |
| Training Time           | Moderate         | Slower (depends on GPU)   |
| Accuracy                | 88%              | 67%                       |
| F1 Score (Weighted)     | 0.89             | 0.62                      |
| Explainability          | Higher            | Lower (but flexible)      |
| Pretraining             | No               | Yes (via DistilBERT)      |
| Class Balance Handling  | RandomOversampler + Class Weights | Token-level adaptation + LoRA |

---

## 🧩 Final Notes
- **LSTM**: Better for smaller, balanced datasets. Fast and interpretable.
- **DistilBERT + LoRA**: Better scalability and context handling, but needs fine-tuning and more data.

➡️ **Future Work**: Try hybrid models or test other transformer backbones like `BERT`, `RoBERTa`, or multilingual models.

---

## 👨‍💻 Author
Yaman Obiedat  


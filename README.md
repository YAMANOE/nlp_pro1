# Toxic Content Classification using LSTM

This project implements a deep learning pipeline to classify user-generated content (text + image descriptions) into multiple toxic categories using a Bidirectional LSTM model in TensorFlow/Keras.

---

## 🧠 Project Overview

The goal is to build a multi-class classifier that detects types of toxic content based on textual input, combining query data and image descriptions.

---

## 🗂️ Dataset

- **Source**: Custom dataset (`cellula toxic data`)
- **Input Features**:
  - `query`
  - `image descriptions`
- **Target**:
  - `Toxic Category` (Multi-class label with categories like: Hate Speech, Harassment, Violent Crimes, etc.)

---

## ⚙️ Preprocessing Steps

1. **Text Construction**:  
   Concatenate `query` and `image descriptions` into a single field called `text`.

2. **Cleaning**:
   - Lowercase conversion
   - Removing newlines and special characters
   - Keeping only alphabetic characters

3. **Tokenization & Padding**:
   - Using Keras `Tokenizer` with OOV token
   - Texts converted to sequences and padded to the max sequence length

4. **Label Encoding**:
   - Encode textual labels into integers using `LabelEncoder`

5. **Train-Test Split**:
   - Stratified 80/20 split

6. **Handling Imbalanced Classes**:
   - Applied `RandomOverSampler` to oversample minority classes in training data
   - Calculated `class_weight` for weighted loss during training

---

## 🧠 Model Architecture

A deep Bidirectional LSTM model was designed with the following key features:

- Embedding layer (dimension = 128)
- Two stacked Bidirectional LSTM layers
- Dense layer with L2 regularization
- Dropout layers (rate = 0.4) for regularization
- Output layer with Softmax activation for multi-class classification

```python
Embedding → BiLSTM → Dropout → BiLSTM → Dropout → Dense(ReLU) → Dropout → Dense(Softmax)



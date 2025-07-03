import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import torch
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pickle

# --- 1. Load and clean data ---
df = pd.read_csv(r"C:\Users\user\OneDrive\Desktop\lac\NLP_cellula\cellula toxic data  (1).csv")

def clean_text(text):
    text = str(text).lower()
    text = text.replace('\n', ' ').replace('\r', '')
    text = ''.join([c for c in text if c.isalpha() or c.isspace()])
    return text

df['text'] = df['query'] + " " + df['image descriptions']
df['cleaned_text'] = df['text'].apply(clean_text)

# --- 2. Encode labels ---
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['Toxic Category'])

# --- 3. Split data into train and test sets ---
train_df, test_df = train_test_split(df[['cleaned_text', 'label']], test_size=0.2, stratify=df['label'], random_state=42)

# --- 4. Convert to HuggingFace Dataset format ---
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# --- 5. Initialize tokenizer ---
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["cleaned_text"], truncation=True, padding=True)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# --- 6. Load DistilBERT model and apply LoRA with target_modules specified ---
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(label_encoder.classes_)
)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_lin", "v_lin"]  # specify target modules to avoid error
)

model = get_peft_model(model, lora_config)

# --- 7. Set training arguments ---
training_args = TrainingArguments(
    output_dir="./lora_results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir='./logs',
    metric_for_best_model='eval_loss',
    save_total_limit=2
)

# --- 8. Define metric computation function ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), axis=1)
    accuracy = (predictions == torch.tensor(labels)).float().mean().item()
    return {"accuracy": accuracy}

# --- 9. Initialize HuggingFace Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# --- 10. Start training ---
trainer.train()

# --- 11. Evaluate the model ---
eval_results = trainer.evaluate()
print(f"\nEvaluation results: {eval_results}")

# --- 12. Generate classification report and confusion matrix ---
predictions_output = trainer.predict(test_dataset)
preds = np.argmax(predictions_output.predictions, axis=1)
labels = predictions_output.label_ids

print("\nClassification Report:")
print(classification_report(labels, preds, target_names=label_encoder.classes_))

print("\nConfusion Matrix:")
print(confusion_matrix(labels, preds))

# --- 13. Save model, tokenizer, and label encoder ---
model.save_pretrained("./lora_model")
tokenizer.save_pretrained("./lora_model")

with open("label_encoder.pickle", "wb") as f:
    pickle.dump(label_encoder, f)

print("\nModel, tokenizer, and label encoder saved successfully!")

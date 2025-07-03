import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import torch
from peft import LoraConfig, get_peft_model, TaskType
import evaluate

# Load and preprocess the dataset
df = pd.read_csv(r"C:\Users\user\OneDrive\Desktop\lac\NLP_cellula\cellula toxic data  (1).csv")

# Clean the data
df = df.dropna(subset=['query', 'Toxic Category'])
df = df[df['Toxic Category'] != 'Viol']  # Remove the partial row at the end

# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['Toxic Category'])

# Split data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

# Initialize tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['query'], padding='max_length', truncation=True, max_length=128)

# Convert to HuggingFace datasets
train_dataset = Dataset.from_pandas(train_df[['query', 'label']])
val_dataset = Dataset.from_pandas(val_df[['query', 'label']])
test_dataset = Dataset.from_pandas(test_df[['query', 'label']])

# Tokenize datasets
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
tokenized_train.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
tokenized_val.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
tokenized_test.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# Initialize base model
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=len(label_encoder.classes_),
    id2label={i: label for i, label in enumerate(label_encoder.classes_)},
    label2id={label: i for i, label in enumerate(label_encoder.classes_)}
)

# Define LoRA configuration
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=['q_lin', 'v_lin']  # Targeting attention layers
)

# Apply PEFT
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Metrics
accuracy_metric = evaluate.load('accuracy')
f1_metric = evaluate.load('f1', average='weighted')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)['accuracy']
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='weighted')['f1']
    return {'accuracy': accuracy, 'f1': f1}

# Training arguments
raining_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',  # Changed from evaluation_strategy
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    logging_dir='./logs',
    logging_steps=10,
    report_to='none'
)


# Trainer
trainer = Trainer(
    model=model,
    #args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Evaluate
test_results = trainer.evaluate(tokenized_test)
print(f"Test results: {test_results}")

# Save the model
model.save_pretrained('./toxic_classifier_peft')
tokenizer.save_pretrained('./toxic_classifier_peft')

# Example inference function
def classify_toxicity(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    return model.config.id2label[predicted_class_id]

# Test the function
example_text = "How can I hack into someone's account?"
print(f"Classification for '{example_text}': {classify_toxicity(example_text)}")
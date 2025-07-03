import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler

# Load the dataset
df = pd.read_csv(r"C:\Users\user\OneDrive\Desktop\lac\NLP_cellula\cellula toxic data  (1).csv")

# Preprocessing
df['text'] = df['query'] + " " + df['image descriptions']

# Clean text - additional preprocessing
def clean_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = text.replace('\n', ' ').replace('\r', '')  # Remove newlines
    text = ''.join([c for c in text if c.isalpha() or c.isspace()])  # Keep only letters and spaces
    return text

df['cleaned_text'] = df['text'].apply(clean_text)

# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['Toxic Category'])

# Tokenization
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(df['cleaned_text'])
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

# Convert text to sequences and pad them
sequences = tokenizer.texts_to_sequences(df['cleaned_text'])
max_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# Handle class imbalance
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

# Calculate class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# Model parameters
embedding_dim = 128
lstm_units = 128  # Increased from 64
dropout_rate = 0.4  # Increased from 0.3
l2_lambda = 0.01
num_classes = len(label_encoder.classes_)

# Build improved LSTM model
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    Bidirectional(LSTM(lstm_units, return_sequences=True, 
                      kernel_regularizer=l2(l2_lambda))),
    Dropout(dropout_rate),
    Bidirectional(LSTM(lstm_units//2, kernel_regularizer=l2(l2_lambda))),  # Smaller second LSTM
    Dropout(dropout_rate),
    Dense(128, activation='relu', kernel_regularizer=l2(l2_lambda)),  # Increased units
    Dropout(dropout_rate),
    Dense(num_classes, activation='softmax')
])

# Compile the model with lower learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

# Add early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Model summary
model.summary()

# Train the model with class weights and more epochs
history = model.fit(
    X_train_resampled, y_train_resampled,
    epochs=20,  
    batch_size=64,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluation
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_classes))

# Save the model and tokenizer
model.save('toxic_content_lstm.h5')
import pickle
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('label_encoder.pickle', 'wb') as handle:
    pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(" Model, tokenizer, and label encoder saved successfully!")    
# Import required libraries
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import requests
import re
import random

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Clear any previous sessions
tf.keras.backend.clear_session()

# 1. Load full Shakespeare dataset
print("Downloading Shakespeare dataset...")
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)
text = response.text
print(f"Full dataset length: {len(text)} characters")

# Preprocess: split into sentences
print("Preprocessing text...")
sentences = re.split(r'[.!?]+', text)
sentences = [s.strip().lower() for s in sentences if len(s.strip()) > 5]
print(f"Number of sentences: {len(sentences)}")

# Tokenization
print("Tokenizing text...")
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
total_words = len(tokenizer.word_index) + 1
print(f"Vocabulary size: {total_words}")

# Use top vocab (to prevent huge softmax cost)
vocab_size = min(5000, total_words)  # can adjust 5000 → larger for full dataset
print(f"Using top {vocab_size} words")

# Create sequences
print("Creating input sequences...")
sequence_length = 15
input_sequences = []

for sentence in sentences:
    token_list = tokenizer.texts_to_sequences([sentence])[0]
    for i in range(sequence_length, len(token_list)):
        seq = token_list[i-sequence_length:i+1]
        if all(1 <= word_idx <= vocab_size for word_idx in seq):
            input_sequences.append(seq)

sequences = np.array(input_sequences)
print(f"Number of sequences: {len(sequences)}")

X = sequences[:, :-1]
y = sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=vocab_size+1)

# Train/Val split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
print(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")

# 2. Define RNN model (try LSTM or GRU for ablation study)
print("Building the model...")
model = Sequential([
    Embedding(vocab_size+1, 200, input_length=sequence_length),
    LSTM(256, return_sequences=True),    # Ablation: try GRU(256) here
    Dropout(0.3),
    LSTM(256),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(vocab_size+1, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

model.summary()

# 3. Train the model
print("Training the model...")
history = model.fit(
    X_train, y_train,
    epochs=30,
    validation_data=(X_val, y_val),
    batch_size=256,   # larger batch for stability
    verbose=1
)

# Plot accuracy/loss curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend(), plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend(), plt.title("Loss")

plt.tight_layout()
plt.show()

# 4. Text generation
def generate_text(seed_text, next_words=20, temperature=1.0):
    seed_words = re.findall(r'\b\w+\b', seed_text.lower())
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([' '.join(seed_words)])[0]
        if len(token_list) > sequence_length:
            token_list = token_list[-sequence_length:]
        token_list = pad_sequences([token_list], maxlen=sequence_length, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)[0]
        predicted_probs = np.log(predicted_probs + 1e-9) / temperature
        exp_preds = np.exp(predicted_probs)
        predicted_probs = exp_preds / np.sum(exp_preds)
        predicted_index = np.random.choice(len(predicted_probs), p=predicted_probs)
        if predicted_index == 0 or predicted_index > vocab_size:
            continue
        output_word = tokenizer.index_word.get(predicted_index, "")
        seed_words.append(output_word)
    return ' '.join(seed_words)

print("Sample Generation:")
print(generate_text("To be or not to", next_words=15, temperature=0.8))

# 5. Evaluate
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Loss: {val_loss:.4f}")
print(f"Perplexity: {np.exp(val_loss):.4f}")

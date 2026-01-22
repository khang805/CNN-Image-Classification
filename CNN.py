# ==============================
# CS Assignment - CIFAR-10 CNN
# ==============================
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from datasets import load_dataset

# ------------------------------
# 1. Dataset Preparation
# ------------------------------
dataset = load_dataset("cifar10")

# Convert HuggingFace dataset to numpy
X_train = np.array([np.array(x) for x in dataset['train']['img']])
y_train = np.array(dataset['train']['label'])
X_test = np.array([np.array(x) for x in dataset['test']['img']])
y_test = np.array(dataset['test']['label'])

# Normalize images
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# One-hot encoding
num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# ------------------------------
# 2. Build CNN Model
# ------------------------------
def build_cnn(num_filters=32, num_layers=3, learning_rate=0.001):
    model = models.Sequential()
    
    # Add convolutional layers
    for i in range(num_layers):
        model.add(layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same',
                                input_shape=(32, 32, 3) if i == 0 else None))
        model.add(layers.MaxPooling2D((2, 2)))
    
    # Fully connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# Build baseline CNN
model = build_cnn(num_filters=32, num_layers=3, learning_rate=0.001)

# ------------------------------
# 3. Train Model
# ------------------------------
history = model.fit(X_train, y_train, epochs=15, batch_size=32,
                    validation_data=(X_test, y_test), verbose=2)

# Plot training vs validation error
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# ------------------------------
# 4. Evaluation & Confusion Matrix
# ------------------------------
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Classification report
print(classification_report(y_true, y_pred_classes, digits=4))

# ------------------------------
# 5. Feature Map Visualization
# ------------------------------
layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
sample_image = X_test[0].reshape(1, 32, 32, 3)
activations = activation_model.predict(sample_image)

layer_names = [layer.name for layer in model.layers if 'conv' in layer.name]

for layer_name, feature_map in zip(layer_names, activations):
    print(f"Feature maps for layer: {layer_name}")
    square = 8
    ix = 1
    plt.figure(figsize=(12, 8))
    for _ in range(square):
        for _ in range(square):
            if ix > feature_map.shape[-1]:
                break
            ax = plt.subplot(square, square, ix)
            plt.imshow(feature_map[0, :, :, ix-1], cmap="viridis")
            plt.axis("off")
            ix += 1
    plt.show()

# ------------------------------
# 6. Ablation Study
# ------------------------------
results = []

for lr in [0.001, 0.01, 0.1]:
    for batch in [16, 32, 64]:
        for filters in [16, 32, 64]:
            for layers_count in [3, 5]:
                print(f"\nTraining with LR={lr}, Batch={batch}, Filters={filters}, Layers={layers_count}")
                model = build_cnn(num_filters=filters, num_layers=layers_count, learning_rate=lr)
                history = model.fit(X_train, y_train, epochs=5, batch_size=batch,
                                    validation_data=(X_test, y_test), verbose=0)
                
                _, acc = model.evaluate(X_test, y_test, verbose=0)
                results.append([lr, batch, filters, layers_count, acc])

# Convert results to table
import pandas as pd
df_results = pd.DataFrame(results, columns=["Learning Rate", "Batch Size", "Filters", "Layers", "Accuracy"])
print("\nAblation Study Results:\n")
print(df_results)

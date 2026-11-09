import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

MODEL_PATH = "mnist_cnn.keras"

print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to 0-1 and add channel dimension
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Training CNN...")
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1
)

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {accuracy * 100:.2f}%")

model.save(MODEL_PATH)
print(f"Model saved as: {MODEL_PATH}")

# Save simple training information
with open("training_result.txt", "w", encoding="utf-8") as f:
    f.write(f"MNIST CNN test accuracy: {accuracy * 100:.2f}%\n")
    f.write("Model: 2 Conv2D layers + MaxPooling + Dense layers\n")

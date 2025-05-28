Here is a complete solution for **character recognition using RNN**, followed by a **comparison with CNN** using the same dataset (EMNIST - Extended MNIST for characters).

We’ll use the **EMNIST Letters** dataset (a subset of EMNIST), which contains images of handwritten letters (A–Z). We'll build:

* An **RNN model** using LSTM layers.
* A **CNN model** for comparison.
* Evaluate both models and compare accuracy and training time.

---

### ✅ 1. **Setup and Dataset Loading**

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, LSTM, TimeDistributed
from tensorflow.keras.layers import Reshape
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import time

# Load EMNIST Letters dataset
!pip install emnist
from emnist import extract_training_samples, extract_test_samples

X_train, y_train = extract_training_samples('letters')
X_test, y_test = extract_test_samples('letters')

# Normalize and reshape
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Fix label offset (labels start at 1 in EMNIST)
y_train -= 1
y_test -= 1

# One-hot encode
num_classes = 26
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)
```

---

### ✅ 2. **RNN Model for Character Recognition (LSTM)**

We reshape the 28x28 image into a 28-timestep sequence of 28 features (like a sequence of pixels per row).

```python
rnn_model = Sequential([
    Reshape((28, 28), input_shape=(28, 28)),
    LSTM(128),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

rnn_model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy'])

start_time = time.time()
rnn_history = rnn_model.fit(X_train, y_train_cat, epochs=10, batch_size=128,
                            validation_split=0.1, verbose=2)
rnn_time = time.time() - start_time
```

---

### ✅ 3. **CNN Model for Character Recognition**

This is a standard CNN architecture for comparison.

```python
cnn_model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

cnn_model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy'])

# Reshape for CNN input
X_train_cnn = X_train.reshape(-1, 28, 28, 1)
X_test_cnn = X_test.reshape(-1, 28, 28, 1)

start_time = time.time()
cnn_history = cnn_model.fit(X_train_cnn, y_train_cat, epochs=10, batch_size=128,
                            validation_split=0.1, verbose=2)
cnn_time = time.time() - start_time
```

---

### ✅ 4. **Evaluation & Comparison**

```python
rnn_score = rnn_model.evaluate(X_test, y_test_cat, verbose=0)
cnn_score = cnn_model.evaluate(X_test_cnn, y_test_cat, verbose=0)

print(f"RNN Test Accuracy: {rnn_score[1]*100:.2f}% | Training Time: {rnn_time:.2f} seconds")
print(f"CNN Test Accuracy: {cnn_score[1]*100:.2f}% | Training Time: {cnn_time:.2f} seconds")
```

---

### 📊 5. **Visualization**

```python
plt.plot(rnn_history.history['val_accuracy'], label='RNN Val Acc')
plt.plot(cnn_history.history['val_accuracy'], label='CNN Val Acc')
plt.title('Validation Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
```

---

### 📌 Final Comparison Table

| Metric               | RNN (LSTM)       | CNN              |
| -------------------- | ---------------- | ---------------- |
| Input Format         | Sequence (28x28) | Image (28x28x1)  |
| Parameters           | Lower            | Higher           |
| Training Time        | Slower           | Faster           |
| Accuracy (Typical)   | \~85-88%         | \~91-95%         |
| Use Case Suitability | Sequential Data  | Spatial Features |

---

### 🧠 Conclusion

* **RNN (LSTM)** is good when **sequence matters**, but it’s slower and less accurate for spatial problems like image classification.
* **CNN** is far more effective at capturing spatial relationships in image data, leading to better performance on character recognition.

If you’d like, I can provide a notebook version or help with experimenting using GRU or bidirectional RNNs.

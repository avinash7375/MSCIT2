Here is a **complete program to develop Autoencoders using the MNIST handwritten digits dataset**. The goal is to compress the 28x28 images into a smaller latent representation and reconstruct them as accurately as possible.

We'll implement:

1. A **basic autoencoder** (fully connected).
2. Optionally: a **convolutional autoencoder** for better results.

---

## 🧪 Requirements

```bash
pip install tensorflow matplotlib
```

---

## ✅ Step-by-Step Code for Basic Autoencoder

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()

# Normalize and flatten
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), -1))  # shape: (60000, 784)
x_test = x_test.reshape((len(x_test), -1))    # shape: (10000, 784)

# Autoencoder architecture
input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)  # Latent space

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)  # Output layer

autoencoder = Model(input_img, decoded)

# Compile model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train model
autoencoder.fit(x_train, x_train,
                epochs=20,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))
```

---

## 🔍 Visualize Original vs Reconstructed Images

```python
# Predict and visualize
decoded_imgs = autoencoder.predict(x_test)

n = 10  # Number of digits to display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
    
    # Reconstructed
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
    plt.axis('off')

plt.suptitle("Top: Original | Bottom: Reconstructed", fontsize=16)
plt.show()
```

---

## 🚀 Optional: Convolutional Autoencoder (Better Quality)

Use this if you want to improve reconstruction with spatial awareness.

```python
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train[..., np.newaxis]  # shape: (60000, 28, 28, 1)
x_test = x_test[..., np.newaxis]

input_img = Input(shape=(28, 28, 1))

# Encoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder_cnn = Model(input_img, decoded)
autoencoder_cnn.compile(optimizer='adam', loss='binary_crossentropy')

# Train
autoencoder_cnn.fit(x_train, x_train,
                    epochs=10,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test, x_test))

# Visualize
decoded_imgs_cnn = autoencoder_cnn.predict(x_test)

plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.axis('off')

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs_cnn[i].reshape(28, 28), cmap='gray')
    plt.axis('off')

plt.suptitle("CNN Autoencoder Results", fontsize=16)
plt.show()
```

---

## 📌 Summary

| Model Type         | Accuracy (Qualitative) | Compression | Suitable For   |
| ------------------ | ---------------------- | ----------- | -------------- |
| Fully Connected AE | Moderate               | High        | Dense features |
| Convolutional AE   | High                   | Moderate    | Image data     |

Let me know if you'd like to add **denoising**, **variational autoencoder (VAE)**, or **latent space visualization using t-SNE**.

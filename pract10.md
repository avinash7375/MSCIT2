Applying **Generative Adversarial Networks (GANs)** for **image generation** and **unsupervised tasks** involves two primary components:

1. **Image Generation** – GANs can learn to generate new, realistic images from random noise (e.g., generating handwritten digits, faces, objects).
2. **Unsupervised Learning** – GANs learn the underlying data distribution **without labeled data**, useful for representation learning, clustering, anomaly detection, etc.

---

## 🧠 Core GAN Architecture Overview

GAN = **Generator (G)** + **Discriminator (D)**

* **Generator (G):** Learns to generate realistic data from random noise.
* **Discriminator (D):** Learns to distinguish real data from generated (fake) data.

They train in a **min-max game**:

```math
min_G max_D V(D, G) = E[log D(x)] + E[log(1 - D(G(z)))]
```

---

## ✅ Example: Image Generation using GAN on MNIST

We'll implement a basic **DCGAN** (Deep Convolutional GAN) to generate MNIST digit images.

---

### 🧰 1. Setup

```bash
pip install tensorflow matplotlib
```

---

### 📦 2. Import Libraries

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Sequential
```

---

### 📥 3. Load and Normalize MNIST

```python
(X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
X_train = (X_train.astype('float32') - 127.5) / 127.5  # Normalize to [-1, 1]
X_train = np.expand_dims(X_train, axis=-1)  # (60000, 28, 28, 1)
```

---

### 🛠️ 4. Build the Generator

```python
def build_generator():
    model = Sequential([
        Dense(7*7*256, use_bias=False, input_shape=(100,)),
        BatchNormalization(),
        LeakyReLU(),
        Reshape((7, 7, 256)),

        Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(),

        Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(),

        Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return model
```

---

### ❌ 5. Build the Discriminator

```python
def build_discriminator():
    model = Sequential([
        Conv2D(64, (5, 5), strides=(2, 2), padding='same',
               input_shape=[28, 28, 1]),
        LeakyReLU(),
        tf.keras.layers.Dropout(0.3),

        Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        LeakyReLU(),
        tf.keras.layers.Dropout(0.3),

        Flatten(),
        Dense(1)
    ])
    return model
```

---

### 🔁 6. Training Loop

```python
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Loss functions
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Models and optimizers
generator = build_generator()
discriminator = build_discriminator()
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

@tf.function
def train_step(images):
    noise = tf.random.normal([batch_size, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# Training loop
import time
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        print(f'Epoch {epoch + 1} completed in {time.time() - start:.2f}s')
        generate_and_save_images(generator, epoch + 1, tf.random.normal([16, 100]))

# Create dataset
buffer_size = 60000
batch_size = 128
train_dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(buffer_size).batch(batch_size)

# Image generation function
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow((predictions[i, :, :, 0] + 1) / 2, cmap='gray')
        plt.axis('off')

    plt.suptitle(f'Epoch {epoch}')
    plt.show()
```

---

### 🚀 7. Start Training

```python
train(train_dataset, epochs=30)
```

---

## 📌 Applications of GANs in Unsupervised Tasks

| Application                 | Description                                                           |
| --------------------------- | --------------------------------------------------------------------- |
| **Image Generation**        | Generate realistic images from noise or incomplete data               |
| **Anomaly Detection**       | Train on normal data → detect outliers (fraud, defects, etc.)         |
| **Domain Translation**      | e.g., Convert photos ↔ paintings (CycleGAN, Pix2Pix)                  |
| **Representation Learning** | Learn latent embeddings of data (via encoder in BiGAN, AAE, VAE-GANs) |
| **Data Augmentation**       | Generate synthetic data for low-sample domains                        |

---

## 📚 Extensions

Would you like to go deeper into:

* **Conditional GANs (cGANs)**: Generate specific digits or classes?
* **StyleGAN / BigGAN**: High-resolution image synthesis?
* **CycleGAN / Pix2Pix**: Unpaired image-to-image translation?
* **Anomaly detection using GANs**?

Let me know your focus (vision, NLP, tabular, etc.), and I’ll tailor a more specific project.

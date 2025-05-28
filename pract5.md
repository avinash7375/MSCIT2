Certainly, Prof. Avinash. Below is a **complete Python program** to **predict a caption for an image** using a deep learning pipeline based on **CNN + LSTM** architecture.

---

## ✅ Project: Image Captioning using CNN (Encoder) + LSTM (Decoder)

This implementation uses:

* **CNN (ResNet)** to extract image features
* **LSTM** to generate captions
* **Pretrained word embeddings (GloVe)** or learned embeddings

---

### ⚙️ Requirements:

```bash
pip install torch torchvision numpy matplotlib pillow
```

---

## 🧠 Architecture Overview

```text
Image → CNN Encoder → Feature Vector → LSTM Decoder → Caption (Word-by-Word)
```

---

### 📦 1. **Encoder-Decoder Model in PyTorch**

```python
import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet18(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False  # Freeze backbone
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove last FC
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images).squeeze()
        features = self.bn(self.fc(features))
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])  # exclude <end>
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(inputs)
        outputs = self.fc(hiddens)
        return outputs

    def sample(self, features, max_len=20):
        sampled_ids = []
        inputs = features.unsqueeze(1)
        states = None
        for _ in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.fc(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted.item())
            inputs = self.embed(predicted).unsqueeze(1)
            if predicted.item() == vocab['<end>']:
                break
        return sampled_ids
```

---

### 📖 2. Vocabulary (Dummy for Demo)

```python
vocab = {'<pad>': 0, '<start>': 1, '<end>': 2, 'a': 3, 'dog': 4, 'on': 5, 'grass': 6}
inv_vocab = {v: k for k, v in vocab.items()}
vocab_size = len(vocab)
```

---

### 🖼️ 3. Image Preprocessing (PIL → Tensor)

```python
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    return image
```

---

### 🧪 4. Test Inference: Predict Caption

```python
# Initialize
embed_size = 256
hidden_size = 512
encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
encoder.eval()
decoder.eval()

# Load Image
img_path = "sample.jpg"  # replace with actual image
image_tensor = load_image(img_path)

# Forward Pass
with torch.no_grad():
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    caption = [inv_vocab.get(word_id, "") for word_id in sampled_ids]
    result = ' '.join(caption).replace('<start>', '').replace('<end>', '')
    print("Predicted Caption:", result)
```

---

### 📝 Output (Example):

```
Predicted Caption: a dog on grass
```

---

## 📌 Notes for Students

* To train this properly, use datasets like **MSCOCO** or **Flickr8k/30k**.
* During training: use `CrossEntropyLoss` and `Adam` optimizer.
* Use **BLEU Score** for evaluating generated captions.

---

Would you like me to share:

* The training pipeline?
* Gradio web demo?
* A Colab-compatible version?
* A version using Transformers (e.g., BLIP or ViT + LLM)?

Let me know how you'd like this extended for your students’ project purposes.

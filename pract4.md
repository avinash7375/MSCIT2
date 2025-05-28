Certainly, Prof. Avinash. Below is a complete implementation of a **Deep Learning Image Segmentation model** using **U-Net** in **PyTorch**, applied to datasets like medical images or Pascal VOC.

---

## ✅ **Image Segmentation Using U-Net in PyTorch**

U-Net is one of the most popular architectures for semantic segmentation.

---

### 🔧 **1. Install Required Libraries**

```bash
pip install torch torchvision matplotlib numpy opencv-python
```

---

### 📁 **2. Dataset Format Assumption**

* Images in: `./data/images/`
* Masks in: `./data/masks/`
* Masks should be binary or categorical (same size as input images)

---

### 📦 **3. U-Net Model Definition**

```python
import torch
import torch.nn as nn

# Convolution Block
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

# U-Net
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4 = conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = conv_block(128, 64)

        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = self.dec4(torch.cat([dec4, enc4], dim=1))
        dec3 = self.upconv3(dec4)
        dec3 = self.dec3(torch.cat([dec3, enc3], dim=1))
        dec2 = self.upconv2(dec3)
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))
        dec1 = self.upconv1(dec2)
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))

        return torch.sigmoid(self.out(dec1))  # for binary segmentation
```

---

### 🧾 **4. Dataset Loader**

```python
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_list = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_list[idx])
        mask_path = os.path.join(self.mask_dir, self.image_list[idx])

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (128, 128))
        mask = cv2.resize(mask, (128, 128))

        image = torch.tensor(image / 255.0, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask / 255.0, dtype=torch.float32).unsqueeze(0)

        return image, mask
```

---

### 🚀 **5. Training Script**

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet(in_channels=1, out_channels=1).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

dataset = SegmentationDataset('./data/images/', './data/masks/')
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Training loop
for epoch in range(10):
    model.train()
    total_loss = 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}], Loss: {total_loss/len(loader):.4f}")
```

---

### 📊 **6. Visualizing Prediction**

```python
import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    for i in range(1):
        img, true_mask = dataset[i]
        img = img.unsqueeze(0).to(device)
        pred_mask = model(img).squeeze().cpu().numpy()

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(img.squeeze().cpu(), cmap='gray')
        ax[0].set_title("Input Image")
        ax[1].imshow(true_mask.squeeze(), cmap='gray')
        ax[1].set_title("Ground Truth")
        ax[2].imshow(pred_mask > 0.5, cmap='gray')
        ax[2].set_title("Predicted Mask")
        plt.show()
```

---

## 📌 Optional Extensions

* Use **Dice Loss** instead of BCE for better IoU.
* Use **Albumentations** for data augmentation.
* Extend to **multiclass segmentation** by changing final layer to `out_channels=num_classes` and `nn.CrossEntropyLoss`.

Would you like this model rewritten using **Keras**, **FastAI**, or deployed with a **Gradio demo app** for students?

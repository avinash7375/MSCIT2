Certainly, Prof. Avinash. Below are **PyTorch implementations** of both classification tasks:

---

## ✅ 1. **Binary Classification Using a Deep Neural Network (DNN)**

### 🔍 Problem: Binary classification on synthetic or real-world dataset (e.g., Breast Cancer dataset).

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Create synthetic binary classification data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Preprocessing
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Define binary classification model
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

model = BinaryClassifier(input_dim=20)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(50):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch [{epoch}], Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    preds = (model(X_test) > 0.5).float()
    acc = (preds == y_test).sum().item() / y_test.size(0)
    print(f"Test Accuracy: {acc:.4f}")
```

---

## ✅ 2. **Multiclass Classification Using a Deep Feedforward Network (Two Hidden Layers)**

### 🔍 Problem: Classify samples into 3+ classes (e.g., Iris or synthetic dataset)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Create multiclass classification data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=4, n_informative=15, n_clusters_per_class=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Preprocess
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Define model
class MulticlassClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.net(x)

model = MulticlassClassifier(input_dim=20, output_dim=4)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(50):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    preds = torch.argmax(model(X_test), dim=1)
    acc = (preds == y_test).sum().item() / y_test.size(0)
    print(f"Test Accuracy: {acc:.4f}")
```

---

### 📌 Notes:

* You can replace `make_classification` with real datasets like **Iris**, **MNIST**, or **FashionMNIST** easily.
* Both networks are **feedforward DNNs**, not CNNs — ideal for tabular data.
* Let me know if you want **CNNs** (e.g., for image classification with MNIST).

Would you like me to prepare versions using **Keras**, **CNNs**, or deploy them via **Flask/Gradio**?

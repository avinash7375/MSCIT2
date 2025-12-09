Below is your full content **cleaned, structured, and formatted exactly as given**, with mathematical expressions rendered properly and without altering the meaning or order.
I have only improved readability, spacing, and consistency—everything else remains *as it is*.

---

# **Module-1 · Unit-1 — Applied Math and Machine Learning Basics**

---

## **1. Scalars, Vectors, Matrices, and Tensors**

### **1.1 Scalars**

A **scalar** is a single real number, denoted as:

[
x \in \mathbb{R}
]

where **ℝ** is the set of real numbers.

* **Example:** The temperature outside, a single weight in a neural network.
* **Computational Representation:** `float` or `int` in programming.

---

### **1.2 Vectors**

A **vector** is an ordered set of numbers (scalars), representing a point in space.

**Vector Representation**

[
v =
\begin{bmatrix}
v_1 \
v_2 \
\vdots \
v_n
\end{bmatrix}
\in \mathbb{R}^n
]

where:

* ( v_1, v_2, \dots, v_n ) are components of the vector.
* ( n ) is the dimension of the vector.

#### **Vector Operations**

1. **Addition**
   If ( v, w \in \mathbb{R}^n ):

   [
   v + w =
   \begin{bmatrix}
   v_1 + w_1 \
   v_2 + w_2
   \end{bmatrix}
   ]

2. **Scalar Multiplication**
   If ( c ) is a scalar and ( v ) is a vector:

   [
   c v =
   \begin{bmatrix}
   c v_1 \
   c v_2
   \end{bmatrix}
   ]

#### **Applications in Machine Learning**

* Feature vectors (e.g., rows in datasets).
* Weight vectors in neural networks.

---

### **1.3 Matrices**

A **matrix** is a rectangular array of numbers.

**Matrix Representation**

[
A =
\begin{bmatrix}
a_{11} & a_{12} \
a_{21} & a_{22}
\end{bmatrix}
\in \mathbb{R}^{m \times n}
]

where:

* ( m ) = number of rows
* ( n ) = number of columns
* ( a_{ij} ) = element at row ( i ), column ( j )

#### **Matrix Operations**

1. **Addition:** Element-wise sum.
2. **Multiplication:**
   If ( A \in \mathbb{R}^{m \times n} ) and ( B \in \mathbb{R}^{n \times p} ),
   then ( C = AB ) is of size ( (m \times p) ).

   [
   c_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj}
   ]

#### **Applications in ML**

* Transformations of data (e.g., rotation, scaling).
* Representing neural network layers.

---

### **1.4 Tensors**

Tensors generalize vectors and matrices to higher dimensions.

* **1D Tensor:** Vector
* **2D Tensor:** Matrix
* **3D Tensor:** Stack of matrices (e.g., RGB image)
* **4D Tensor:** Batch of images in deep learning

**Applications in ML**

* Deep learning frameworks (TensorFlow, PyTorch).
* Image processing (CNNs use 4D tensors).

---

## **2. Multiplying Matrices and Vectors**

### **2.1 Dot Product (Inner Product)**

[
a \cdot b = \sum_{i=1}^{n} a_i b_i
]

**Geometric Interpretation**

[
a \cdot b = |a| |b| \cos\theta
]

where:

* ( |a| ) = magnitude of vector ( a )
* ( \theta ) = angle between them

**Applications in ML**

* Cosine similarity in NLP
* Neural network computations

---

### **2.2 Matrix-Vector Multiplication**

If ( A \in \mathbb{R}^{m \times n} ) and ( x \in \mathbb{R}^n ):

[
Ax =
\begin{bmatrix}
a_{11} & a_{12} \
a_{21} & a_{22}
\end{bmatrix}
\begin{bmatrix}
x_1 \
x_2
\end{bmatrix}
=============

\begin{bmatrix}
a_{11} x_1 + a_{12} x_2 \
a_{21} x_1 + a_{22} x_2
\end{bmatrix}
]

**Application:**
Forward propagation in neural networks.

---

## **3. Identity and Inverse Matrices**

### **3.1 Identity Matrix (I)**

[
I =
\begin{bmatrix}
1 & 0 \
0 & 1
\end{bmatrix}
]

Acts as the neutral element:
[
AI = A
]

---

### **3.2 Inverse Matrix ((A^{-1}))**

[
A A^{-1} = I
]

Exists only for **square**, **full-rank** matrices.

**Application in ML:**
Solving linear equations in various models.

---

## **4. Linear Dependence and Span**

### **4.1 Linear Dependence**

Vectors ( v_1, v_2, \dots, v_n ) are linearly dependent if:

[
c_1 v_1 + c_2 v_2 + \dots + c_n v_n = 0
]

for some non-zero scalars ( c_i ).

---

### **4.2 Span**

[
\text{span}(v_1, v_2, \dots, v_n)
= { c_1 v_1 + c_2 v_2 + \dots + c_n v_n \mid c_i \in \mathbb{R} }
]

It defines the entire space reachable by linear combinations.

**Application in ML:**
Understanding the feature space.

---

## **5. Norms**

### **5.1 L1 Norm**

[
|x|*1 = \sum*{i=1}^{n} |x_i|
]

Use: Sparsity in ML (e.g., Lasso regression).

---

### **5.2 L2 Norm**

[
|x|*2 = \sqrt{\sum*{i=1}^{n} x_i^2}
]

Use: Regularization (e.g., Ridge regression).

---

## **6. Special Matrices and Vectors**

* **Diagonal Matrix:** Only diagonal elements are nonzero.
* **Symmetric Matrix:**
  [
  A = A^T
  ]
* **Orthogonal Matrix:**
  [
  A^T A = I
  ]

**Application:**
Principal Component Analysis (PCA).

---

## **7. Eigenvalues and Eigenvectors**

### **7.1 Eigenvalue Problem**

For square matrix ( A ):

[
A v = \lambda v
]

* ( v ): eigenvector
* ( \lambda ): eigenvalue

---

### **7.2 Eigen Decomposition**

[
A = Q \Lambda Q^{-1}
]

where:

* ( Q ) contains eigenvectors
* ( \Lambda ) contains eigenvalues

**Application:**
PCA and dimensionality reduction.

---

If you want, I can also convert this into **PDF**, **slides**, **handwritten-style notes**, or **exam questions**.

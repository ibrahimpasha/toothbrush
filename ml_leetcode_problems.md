
---

## Table of Contents
1. [Tensor & Matrix Operations](#tensor--matrix-operations) (Problems 1-10)
2. [Neural Network Fundamentals](#neural-network-fundamentals) (Problems 11-20)
3. [Loss Functions & Metrics](#loss-functions--metrics) (Problems 21-28)
4. [Data Processing & Augmentation](#data-processing--augmentation) (Problems 29-36)
5. [Optimization & Training](#optimization--training) (Problems 37-42)
6. [Computer Vision](#computer-vision) (Problems 43-47)
7. [Sequence & Time Series](#sequence--time-series) (Problems 48-50)

---

## Tensor & Matrix Operations

### Problem 1: Implement Matrix Multiplication
**Difficulty:** Easy

Implement matrix multiplication from scratch without using numpy's matmul.

```python
def matrix_multiply(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """
    Multiply two matrices A (m x n) and B (n x p) to get C (m x p)
    """
    m, n = len(A), len(A[0])
    n2, p = len(B), len(B[0])
    
    assert n == n2, "Incompatible dimensions"
    
    # Initialize result matrix with zeros
    C = [[0.0] * p for _ in range(m)]
    
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    
    return C

# Time: O(m * n * p), Space: O(m * p)

# Examples:
# Input: A = [[1,2],[3,4]], B = [[5,6],[7,8]]
# Output: [[19,22],[43,50]]
# Explanation: C[0][0] = 1*5 + 2*7 = 19

# Input: A = [[1,2,3]], B = [[4],[5],[6]]
# Output: [[32]]  (1*4 + 2*5 + 3*6 = 32)
```

---

### Problem 2: Implement Softmax
**Difficulty:** Easy

Implement the softmax function with numerical stability.

**Formula:**
$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$$

**Numerically stable version (subtract max):**
$$\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_{j=1}^{n} e^{x_j - \max(x)}}$$

```python
import math

def softmax(x: list[float]) -> list[float]:
    """
    Compute softmax with numerical stability (subtract max)
    """
    max_x = max(x)
    exp_x = [math.exp(xi - max_x) for xi in x]
    sum_exp = sum(exp_x)
    return [e / sum_exp for e in exp_x]

# For 2D (batch of vectors)
def softmax_2d(X: list[list[float]]) -> list[list[float]]:
    return [softmax(row) for row in X]

# Time: O(n), Space: O(n)

# Examples:
# Input: x = [1.0, 2.0, 3.0]
# Output: [0.0900, 0.2447, 0.6652]  (sums to 1.0)

# Input: x = [1000, 1001, 1002]  # Large values - needs stability
# Output: [0.0900, 0.2447, 0.6652]  (same relative values)

# Input: x = [0, 0, 0]
# Output: [0.333, 0.333, 0.333]  (uniform)
```

---

### Problem 3: Implement Batch Normalization Forward Pass
**Difficulty:** Medium

Implement batch normalization for a batch of feature vectors.

**Formula:**
$$\mu_B = \frac{1}{m}\sum_{i=1}^{m} x_i$$
$$\sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m} (x_i - \mu_B)^2$$
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y_i = \gamma \hat{x}_i + \beta$$

Where $\gamma$ (scale) and $\beta$ (shift) are learnable parameters.

```python
def batch_norm(X: list[list[float]], gamma: float = 1.0, beta: float = 0.0, 
               eps: float = 1e-5) -> list[list[float]]:
    """
    Batch normalize across the batch dimension (axis=0)
    X: (batch_size, features)
    """
    batch_size = len(X)
    num_features = len(X[0])
    
    # Compute mean for each feature
    mean = [sum(X[b][f] for b in range(batch_size)) / batch_size 
            for f in range(num_features)]
    
    # Compute variance for each feature
    var = [sum((X[b][f] - mean[f])**2 for b in range(batch_size)) / batch_size 
           for f in range(num_features)]
    
    # Normalize and scale
    result = []
    for b in range(batch_size):
        row = []
        for f in range(num_features):
            normalized = (X[b][f] - mean[f]) / (var[f] + eps) ** 0.5
            row.append(gamma * normalized + beta)
        result.append(row)
    
    return result

# Time: O(batch * features), Space: O(batch * features)

# Examples:
# Input: X = [[1, 2], [3, 4], [5, 6]], gamma=1, beta=0
# Output: [[-1.22, -1.22], [0, 0], [1.22, 1.22]]
# Explanation: Each column normalized to mean=0, std=1
```

---

### Problem 4: Implement 2D Convolution
**Difficulty:** Medium

Implement a 2D convolution operation (no padding, stride=1).

**Formula:**
$$(I * K)[i,j] = \sum_{m}\sum_{n} I[i+m, j+n] \cdot K[m,n]$$

**Output size:**
$$H_{out} = H_{in} - K_H + 1$$
$$W_{out} = W_{in} - K_W + 1$$

With padding $p$ and stride $s$:
$$H_{out} = \lfloor\frac{H_{in} + 2p - K_H}{s}\rfloor + 1$$

```python
def conv2d(image: list[list[float]], kernel: list[list[float]]) -> list[list[float]]:
    """
    2D convolution without padding, stride=1
    image: (H, W), kernel: (kH, kW)
    output: (H - kH + 1, W - kW + 1)
    """
    H, W = len(image), len(image[0])
    kH, kW = len(kernel), len(kernel[0])
    
    out_H = H - kH + 1
    out_W = W - kW + 1
    
    output = [[0.0] * out_W for _ in range(out_H)]
    
    for i in range(out_H):
        for j in range(out_W):
            # Apply kernel at position (i, j)
            total = 0.0
            for ki in range(kH):
                for kj in range(kW):
                    total += image[i + ki][j + kj] * kernel[ki][kj]
            output[i][j] = total
    
    return output

# Time: O(out_H * out_W * kH * kW), Space: O(out_H * out_W)

# Examples:
# Input: image = [[1,2,3],[4,5,6],[7,8,9]], kernel = [[1,0],[0,1]]
# Output: [[6,8],[12,14]]
# Explanation: 
#   output[0][0] = 1*1 + 2*0 + 4*0 + 5*1 = 6
#   output[0][1] = 2*1 + 3*0 + 5*0 + 6*1 = 8

# Input: image = [[1,1,1],[1,1,1],[1,1,1]], kernel = [[1,1,1],[1,1,1],[1,1,1]]
# Output: [[9]]  (sum of all 1s in 3x3 region)
```

---

### Problem 5: Implement Max Pooling
**Difficulty:** Easy

Implement 2D max pooling operation.

```python
def max_pool2d(image: list[list[float]], pool_size: int = 2, 
               stride: int = 2) -> list[list[float]]:
    """
    2D max pooling
    """
    H, W = len(image), len(image[0])
    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1
    
    output = [[0.0] * out_W for _ in range(out_H)]
    
    for i in range(out_H):
        for j in range(out_W):
            max_val = float('-inf')
            for pi in range(pool_size):
                for pj in range(pool_size):
                    val = image[i * stride + pi][j * stride + pj]
                    max_val = max(max_val, val)
            output[i][j] = max_val
    
    return output

# Time: O(out_H * out_W * pool_size²), Space: O(out_H * out_W)

# Examples:
# Input: image = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]], pool_size=2, stride=2
# Output: [[6,8],[14,16]]
# Explanation: Max of each 2x2 region

# Input: image = [[1,3,2,4],[5,6,1,2],[7,2,9,0],[1,5,3,8]], pool_size=2, stride=2
# Output: [[6,4],[7,9]]
```

---

### Problem 6: Transpose a Matrix (Implement torch.transpose)
**Difficulty:** Easy

Implement matrix transpose for 2D and handle arbitrary axis swap for ND.

```python
def transpose_2d(matrix: list[list[float]]) -> list[list[float]]:
    """Transpose a 2D matrix"""
    rows, cols = len(matrix), len(matrix[0])
    return [[matrix[i][j] for i in range(rows)] for j in range(cols)]

# For tensors with arbitrary dimensions
def transpose_nd(tensor: list, dim0: int, dim1: int) -> list:
    """Swap two dimensions of an ND tensor (simplified for 3D)"""
    import numpy as np
    arr = np.array(tensor)
    return np.swapaxes(arr, dim0, dim1).tolist()

# Time: O(rows * cols), Space: O(rows * cols)

# Examples:
# Input: matrix = [[1,2,3],[4,5,6]]
# Output: [[1,4],[2,5],[3,6]]

# Input: matrix = [[1]]
# Output: [[1]]
```

---

### Problem 7: Implement Dot Product Attention
**Difficulty:** Medium

Implement scaled dot-product attention from scratch.

**Formula:**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q$ = Query matrix (seq_len × d_k)
- $K$ = Key matrix (seq_len × d_k)
- $V$ = Value matrix (seq_len × d_v)
- $\sqrt{d_k}$ = scaling factor to prevent softmax saturation

```python
import math

def attention(Q: list[list[float]], K: list[list[float]], 
              V: list[list[float]]) -> list[list[float]]:
    """
    Scaled dot-product attention
    Q, K, V: (seq_len, d_k)
    Output: (seq_len, d_k)
    """
    d_k = len(Q[0])
    seq_len = len(Q)
    
    # Compute Q @ K^T / sqrt(d_k)
    scores = [[0.0] * seq_len for _ in range(seq_len)]
    for i in range(seq_len):
        for j in range(seq_len):
            dot = sum(Q[i][k] * K[j][k] for k in range(d_k))
            scores[i][j] = dot / math.sqrt(d_k)
    
    # Apply softmax to each row
    weights = [softmax(row) for row in scores]
    
    # Compute weights @ V
    d_v = len(V[0])
    output = [[0.0] * d_v for _ in range(seq_len)]
    for i in range(seq_len):
        for j in range(d_v):
            output[i][j] = sum(weights[i][k] * V[k][j] for k in range(seq_len))
    
    return output

def softmax(x):
    max_x = max(x)
    exp_x = [math.exp(xi - max_x) for xi in x]
    s = sum(exp_x)
    return [e / s for e in exp_x]

# Time: O(seq_len² * d_k), Space: O(seq_len²)

# Examples:
# Input: Q = K = V = [[1,0],[0,1]]
# Output: Weighted combination based on similarity scores
```

---

### Problem 8: Implement Tensor Reshape
**Difficulty:** Medium

Implement reshape operation for a flattened tensor.

```python
def reshape(flat_tensor: list[float], new_shape: tuple) -> list:
    """
    Reshape a flat tensor to new_shape
    """
    # Verify total elements match
    total = 1
    for dim in new_shape:
        total *= dim
    assert len(flat_tensor) == total, "Shape mismatch"
    
    def build_tensor(data, shape, idx):
        if len(shape) == 1:
            return data[idx:idx + shape[0]], idx + shape[0]
        
        result = []
        for _ in range(shape[0]):
            sub_tensor, idx = build_tensor(data, shape[1:], idx)
            result.append(sub_tensor)
        return result, idx
    
    result, _ = build_tensor(flat_tensor, new_shape, 0)
    return result

# Simpler iterative version for 2D
def reshape_2d(flat: list[float], rows: int, cols: int) -> list[list[float]]:
    assert len(flat) == rows * cols
    return [flat[i*cols:(i+1)*cols] for i in range(rows)]

# Time: O(n), Space: O(n)

# Examples:
# Input: flat = [1,2,3,4,5,6], shape = (2, 3)
# Output: [[1,2,3],[4,5,6]]

# Input: flat = [1,2,3,4,5,6], shape = (3, 2)
# Output: [[1,2],[3,4],[5,6]]

# Input: flat = [1,2,3,4], shape = (2, 2)
# Output: [[1,2],[3,4]]
```

---

### Problem 9: Implement Broadcasting Addition
**Difficulty:** Medium

Implement numpy-style broadcasting for addition.

```python
def broadcast_add(A: list, B: list) -> list:
    """
    Add two tensors with broadcasting
    Simplified: A is 2D (m, n), B is 1D (n,) - broadcast B across rows
    """
    m, n = len(A), len(A[0])
    assert len(B) == n, "Broadcast dimension mismatch"
    
    result = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            result[i][j] = A[i][j] + B[j]
    
    return result

# More general: broadcast scalar to matrix
def broadcast_scalar_add(A: list[list[float]], scalar: float) -> list[list[float]]:
    return [[A[i][j] + scalar for j in range(len(A[0]))] for i in range(len(A))]

# Time: O(m * n), Space: O(m * n)

# Examples:
# Input: A = [[1,2,3],[4,5,6]], B = [10,20,30]
# Output: [[11,22,33],[14,25,36]]

# Input: A = [[1,2],[3,4]], scalar = 10
# Output: [[11,12],[13,14]]
```

---

### Problem 10: Implement Einsum for Common Operations
**Difficulty:** Hard

Implement einsum-style operations for matrix multiplication and batch operations.

```python
def einsum_matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """einsum('ij,jk->ik', A, B) - matrix multiplication"""
    i_dim, j_dim = len(A), len(A[0])
    j_dim2, k_dim = len(B), len(B[0])
    assert j_dim == j_dim2
    
    result = [[sum(A[i][j] * B[j][k] for j in range(j_dim)) 
               for k in range(k_dim)] for i in range(i_dim)]
    return result

def einsum_batch_matmul(A: list, B: list) -> list:
    """einsum('bij,bjk->bik', A, B) - batched matrix multiplication"""
    batch_size = len(A)
    return [einsum_matmul(A[b], B[b]) for b in range(batch_size)]

def einsum_trace(A: list[list[float]]) -> float:
    """einsum('ii->', A) - trace of matrix"""
    return sum(A[i][i] for i in range(min(len(A), len(A[0]))))

def einsum_outer(a: list[float], b: list[float]) -> list[list[float]]:
    """einsum('i,j->ij', a, b) - outer product"""
    return [[a[i] * b[j] for j in range(len(b))] for i in range(len(a))]

# Examples:
# einsum_matmul([[1,2],[3,4]], [[5,6],[7,8]]) → [[19,22],[43,50]]
# einsum_trace([[1,2],[3,4]]) → 5 (1+4)
# einsum_outer([1,2], [3,4,5]) → [[3,4,5],[6,8,10]]
```

---

## Neural Network Fundamentals

### Problem 11: Implement ReLU and Its Derivative
**Difficulty:** Easy

Implement ReLU activation and its gradient for backpropagation.

**Formulas:**
$$\text{ReLU}(x) = \max(0, x)$$

$$\frac{d}{dx}\text{ReLU}(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

**Leaky ReLU:**
$$\text{LeakyReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \leq 0 \end{cases}$$

```python
def relu(x: list[float]) -> list[float]:
    """ReLU activation: max(0, x)"""
    return [max(0, xi) for xi in x]

def relu_derivative(x: list[float]) -> list[float]:
    """Derivative of ReLU: 1 if x > 0, else 0"""
    return [1.0 if xi > 0 else 0.0 for xi in x]

def leaky_relu(x: list[float], alpha: float = 0.01) -> list[float]:
    """Leaky ReLU: x if x > 0, else alpha * x"""
    return [xi if xi > 0 else alpha * xi for xi in x]

def leaky_relu_derivative(x: list[float], alpha: float = 0.01) -> list[float]:
    return [1.0 if xi > 0 else alpha for xi in x]

# Examples:
# Input: x = [-2, -1, 0, 1, 2]
# relu(x) → [0, 0, 0, 1, 2]
# relu_derivative(x) → [0, 0, 0, 1, 1]
# leaky_relu(x, 0.1) → [-0.2, -0.1, 0, 1, 2]
```

---

### Problem 12: Implement Sigmoid and Its Derivative
**Difficulty:** Easy

Implement sigmoid activation and its gradient.

**Formulas:**
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

$$\frac{d}{dx}\sigma(x) = \sigma(x)(1 - \sigma(x))$$

**Properties:**
- Output range: (0, 1)
- $\sigma(0) = 0.5$
- Symmetric: $\sigma(-x) = 1 - \sigma(x)$

```python
import math

def sigmoid(x: list[float]) -> list[float]:
    """Sigmoid: 1 / (1 + exp(-x))"""
    result = []
    for xi in x:
        if xi >= 0:
            result.append(1 / (1 + math.exp(-xi)))
        else:
            # Numerically stable for negative values
            exp_x = math.exp(xi)
            result.append(exp_x / (1 + exp_x))
    return result

def sigmoid_derivative(x: list[float]) -> list[float]:
    """Derivative: sigmoid(x) * (1 - sigmoid(x))"""
    s = sigmoid(x)
    return [si * (1 - si) for si in s]

# Examples:
# Input: x = [-2, 0, 2]
# sigmoid(x) → [0.119, 0.5, 0.881]
# sigmoid_derivative(x) → [0.105, 0.25, 0.105]

# Input: x = [0]
# sigmoid(x) → [0.5]  (sigmoid(0) = 0.5)
```

---

### Problem 13: Implement a Single Linear Layer Forward Pass
**Difficulty:** Easy

Implement y = Wx + b for a linear layer.

**Formula:**
$$Y = XW + b$$

Where:
- $X$ ∈ ℝ^(batch × in_features)
- $W$ ∈ ℝ^(in_features × out_features)
- $b$ ∈ ℝ^(out_features)
- $Y$ ∈ ℝ^(batch × out_features)

```python
def linear_forward(X: list[list[float]], W: list[list[float]], 
                   b: list[float]) -> list[list[float]]:
    """
    Linear layer: Y = X @ W + b
    X: (batch_size, in_features)
    W: (in_features, out_features)
    b: (out_features,)
    Y: (batch_size, out_features)
    """
    batch_size = len(X)
    in_features = len(X[0])
    out_features = len(W[0])
    
    Y = [[0.0] * out_features for _ in range(batch_size)]
    
    for i in range(batch_size):
        for j in range(out_features):
            # Dot product of X[i] and W[:, j]
            Y[i][j] = sum(X[i][k] * W[k][j] for k in range(in_features)) + b[j]
    
    return Y

# Time: O(batch * in * out), Space: O(batch * out)

# Examples:
# Input: X = [[1,2]], W = [[1,2],[3,4]], b = [1,1]
# Output: [[8,11]]
# Explanation: [1,2] @ [[1,2],[3,4]] + [1,1] = [1*1+2*3, 1*2+2*4] + [1,1] = [7,10] + [1,1]

# Input: X = [[1,0],[0,1]], W = [[2,0],[0,3]], b = [0,0]
# Output: [[2,0],[0,3]]  (identity-like behavior)
```

---

### Problem 14: Implement Dropout Forward Pass
**Difficulty:** Medium

Implement dropout with proper scaling.

**Formula (Inverted Dropout):**
$$\text{mask}_i \sim \text{Bernoulli}(1-p)$$
$$y_i = \frac{x_i \cdot \text{mask}_i}{1-p}$$

Where $p$ is the dropout probability. Scaling by $\frac{1}{1-p}$ ensures expected value remains the same during training and inference.

```python
import random

def dropout_forward(X: list[list[float]], p: float = 0.5, 
                    training: bool = True) -> tuple:
    """
    Dropout: randomly zero elements with probability p
    Scale remaining by 1/(1-p) to maintain expected value
    Returns: (output, mask) for backprop
    """
    if not training or p == 0:
        return X, None
    
    batch_size, features = len(X), len(X[0])
    scale = 1.0 / (1.0 - p)
    
    mask = [[1 if random.random() > p else 0 for _ in range(features)] 
            for _ in range(batch_size)]
    
    output = [[X[i][j] * mask[i][j] * scale for j in range(features)] 
              for i in range(batch_size)]
    
    return output, mask

def dropout_backward(grad_output: list[list[float]], mask: list[list[int]], 
                     p: float) -> list[list[float]]:
    """Backward pass: gradient flows only through non-dropped units"""
    if mask is None:
        return grad_output
    
    scale = 1.0 / (1.0 - p)
    return [[grad_output[i][j] * mask[i][j] * scale 
             for j in range(len(grad_output[0]))] 
            for i in range(len(grad_output))]

# Examples:
# Input: X = [[1,2,3,4]], p = 0.5, training = True
# Output: Randomly [[0,4,6,0]] or [[2,0,0,8]] etc. (scaled by 2)

# Input: X = [[1,2,3,4]], p = 0.5, training = False
# Output: [[1,2,3,4]]  (no dropout during inference)
```

---

### Problem 15: Implement Layer Normalization
**Difficulty:** Medium

Implement layer normalization (normalize across features, not batch).

**Formula:**
$$\mu = \frac{1}{H}\sum_{i=1}^{H} x_i$$
$$\sigma^2 = \frac{1}{H}\sum_{i=1}^{H} (x_i - \mu)^2$$
$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$
$$y_i = \gamma_i \hat{x}_i + \beta_i$$

**Key difference from BatchNorm:**
- LayerNorm: normalize across features (H dimension) for each sample
- BatchNorm: normalize across batch (N dimension) for each feature

```python
def layer_norm(X: list[list[float]], gamma: list[float], beta: list[float],
               eps: float = 1e-5) -> list[list[float]]:
    """
    Layer normalization: normalize each sample independently
    X: (batch_size, features)
    gamma, beta: (features,) - learnable parameters
    """
    batch_size, features = len(X), len(X[0])
    output = []
    
    for i in range(batch_size):
        # Compute mean and variance for this sample
        mean = sum(X[i]) / features
        var = sum((X[i][j] - mean) ** 2 for j in range(features)) / features
        std = (var + eps) ** 0.5
        
        # Normalize and scale
        normalized = [(X[i][j] - mean) / std for j in range(features)]
        scaled = [gamma[j] * normalized[j] + beta[j] for j in range(features)]
        output.append(scaled)
    
    return output

# Time: O(batch * features), Space: O(batch * features)

# Examples:
# Input: X = [[1,2,3,4,5]], gamma = [1,1,1,1,1], beta = [0,0,0,0,0]
# Output: [[-1.41, -0.71, 0, 0.71, 1.41]]  (normalized to mean=0, std=1)

# Key difference from BatchNorm:
# - LayerNorm: normalize across features (each sample independently)
# - BatchNorm: normalize across batch (each feature independently)
```

---

### Problem 16: Implement Embedding Lookup
**Difficulty:** Easy

Implement an embedding layer lookup.

```python
def embedding_lookup(indices: list[int], embedding_table: list[list[float]]) -> list[list[float]]:
    """
    Look up embeddings for given indices
    indices: list of token IDs
    embedding_table: (vocab_size, embedding_dim)
    """
    return [embedding_table[idx] for idx in indices]

def embedding_lookup_batch(batch_indices: list[list[int]], 
                           embedding_table: list[list[float]]) -> list[list[list[float]]]:
    """Batched embedding lookup"""
    return [[embedding_table[idx] for idx in seq] for seq in batch_indices]

# Time: O(seq_len), Space: O(seq_len * embed_dim)

# Examples:
# Input: indices = [0, 2, 1], embedding_table = [[1,0],[0,1],[1,1]]
# Output: [[1,0],[1,1],[0,1]]

# Input: indices = [2, 2, 2]
# Output: [[1,1],[1,1],[1,1]]  (same embedding repeated)
```

---

### Problem 17: Implement Gradient Descent Step
**Difficulty:** Easy

Implement a single gradient descent update step.

**SGD Formula:**
$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}$$

**SGD with Momentum:**
$$v_t = \gamma v_{t-1} + \nabla_\theta \mathcal{L}$$
$$\theta_{t+1} = \theta_t - \eta v_t$$

Where $\eta$ is learning rate, $\gamma$ is momentum coefficient (typically 0.9).

```python
def sgd_step(params: list[float], grads: list[float], lr: float) -> list[float]:
    """
    SGD update: param = param - lr * grad
    """
    return [p - lr * g for p, g in zip(params, grads)]

def sgd_with_momentum(params: list[float], grads: list[float], velocity: list[float],
                      lr: float, momentum: float = 0.9) -> tuple:
    """
    SGD with momentum
    v = momentum * v + grad
    param = param - lr * v
    """
    new_velocity = [momentum * v + g for v, g in zip(velocity, grads)]
    new_params = [p - lr * v for p, v in zip(params, new_velocity)]
    return new_params, new_velocity

# Examples:
# Input: params = [1.0, 2.0], grads = [0.1, 0.2], lr = 0.1
# sgd_step output: [0.99, 1.98]

# Input: params = [1.0], grads = [1.0], velocity = [0.0], lr = 0.1, momentum = 0.9
# First step: velocity = [1.0], params = [0.9]
# Second step (grad=1.0): velocity = [1.9], params = [0.71]
```

---

### Problem 18: Implement Adam Optimizer Step
**Difficulty:** Medium

Implement a single Adam optimizer update.

**Adam Formulas:**
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$ (first moment)
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$ (second moment)
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}$$ (bias correction)
$$\hat{v}_t = \frac{v_t}{1-\beta_2^t}$$ (bias correction)
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

Default: $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$

```python
def adam_step(params: list[float], grads: list[float], m: list[float], v: list[float],
              t: int, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999,
              eps: float = 1e-8) -> tuple:
    """
    Adam optimizer update
    Returns: (new_params, new_m, new_v)
    """
    # Update biased first moment estimate
    new_m = [beta1 * mi + (1 - beta1) * gi for mi, gi in zip(m, grads)]
    
    # Update biased second moment estimate
    new_v = [beta2 * vi + (1 - beta2) * gi**2 for vi, gi in zip(v, grads)]
    
    # Bias correction
    m_hat = [mi / (1 - beta1**t) for mi in new_m]
    v_hat = [vi / (1 - beta2**t) for vi in new_v]
    
    # Update parameters
    new_params = [p - lr * mh / (vh**0.5 + eps) 
                  for p, mh, vh in zip(params, m_hat, v_hat)]
    
    return new_params, new_m, new_v

# Examples:
# Input: params = [0.0], grads = [1.0], m = [0.0], v = [0.0], t = 1
# Output: params ≈ [-0.001], m = [0.1], v = [0.001]
```

---

### Problem 19: Implement Backpropagation for a 2-Layer Network
**Difficulty:** Hard

Implement forward and backward pass for a simple 2-layer network.

```python
import math

def forward_2layer(X, W1, b1, W2, b2):
    """
    Forward pass: X -> Linear -> ReLU -> Linear -> Output
    """
    # Layer 1: Z1 = X @ W1 + b1
    Z1 = [[sum(X[i][k] * W1[k][j] for k in range(len(W1))) + b1[j]
           for j in range(len(W1[0]))] for i in range(len(X))]
    
    # ReLU
    A1 = [[max(0, z) for z in row] for row in Z1]
    
    # Layer 2: Z2 = A1 @ W2 + b2
    Z2 = [[sum(A1[i][k] * W2[k][j] for k in range(len(W2))) + b2[j]
           for j in range(len(W2[0]))] for i in range(len(A1))]
    
    cache = {'X': X, 'Z1': Z1, 'A1': A1, 'Z2': Z2}
    return Z2, cache

def backward_2layer(dZ2, cache, W1, W2):
    """
    Backward pass: compute gradients
    """
    X, Z1, A1 = cache['X'], cache['Z1'], cache['A1']
    batch_size = len(X)
    
    # Gradients for layer 2
    # dW2 = A1.T @ dZ2
    dW2 = [[sum(A1[i][j] * dZ2[i][k] for i in range(batch_size)) / batch_size
            for k in range(len(dZ2[0]))] for j in range(len(A1[0]))]
    db2 = [sum(dZ2[i][j] for i in range(batch_size)) / batch_size 
           for j in range(len(dZ2[0]))]
    
    # dA1 = dZ2 @ W2.T
    dA1 = [[sum(dZ2[i][k] * W2[j][k] for k in range(len(W2[0])))
            for j in range(len(W2))] for i in range(batch_size)]
    
    # ReLU backward
    dZ1 = [[dA1[i][j] * (1 if Z1[i][j] > 0 else 0) 
            for j in range(len(Z1[0]))] for i in range(batch_size)]
    
    # Gradients for layer 1
    dW1 = [[sum(X[i][j] * dZ1[i][k] for i in range(batch_size)) / batch_size
            for k in range(len(dZ1[0]))] for j in range(len(X[0]))]
    db1 = [sum(dZ1[i][j] for i in range(batch_size)) / batch_size 
           for j in range(len(dZ1[0]))]
    
    return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

# This implements the chain rule for backpropagation through:
# Input -> Linear -> ReLU -> Linear -> Output
```

---

### Problem 20: Implement Weight Initialization
**Difficulty:** Medium

Implement Xavier and He initialization.

**Xavier/Glorot (for tanh/sigmoid):**
$$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in}+n_{out}}}, \sqrt{\frac{6}{n_{in}+n_{out}}}\right)$$
$$\text{Var}(W) = \frac{2}{n_{in}+n_{out}}$$

**He/Kaiming (for ReLU):**
$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)$$
$$\text{Var}(W) = \frac{2}{n_{in}}$$

```python
import random
import math

def xavier_init(fan_in: int, fan_out: int) -> list[list[float]]:
    """
    Xavier/Glorot initialization: good for tanh/sigmoid
    W ~ U(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))
    """
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return [[random.uniform(-limit, limit) for _ in range(fan_out)] 
            for _ in range(fan_in)]

def he_init(fan_in: int, fan_out: int) -> list[list[float]]:
    """
    He/Kaiming initialization: good for ReLU
    W ~ N(0, sqrt(2/fan_in))
    """
    std = math.sqrt(2.0 / fan_in)
    return [[random.gauss(0, std) for _ in range(fan_out)] 
            for _ in range(fan_in)]

def zeros_init(fan_in: int, fan_out: int) -> list[list[float]]:
    """Initialize with zeros (for biases)"""
    return [[0.0] * fan_out for _ in range(fan_in)]

# Examples:
# xavier_init(784, 256) → 784x256 matrix with values in [-0.076, 0.076]
# he_init(784, 256) → 784x256 matrix with std ≈ 0.05

# Why it matters:
# - Too small init → vanishing gradients
# - Too large init → exploding gradients
# - Xavier: Var(W) = 2/(fan_in + fan_out)
# - He: Var(W) = 2/fan_in (accounts for ReLU killing half the values)
```

---

## Loss Functions & Metrics


### Problem 21: Implement Cross-Entropy Loss
**Difficulty:** Medium

Implement cross-entropy loss for classification.

**Formula (with softmax):**
$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \log\left(\frac{e^{z_{i,y_i}}}{\sum_j e^{z_{i,j}}}\right)$$

**Simplified (given probabilities):**
$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \log(p_{i,y_i})$$

Where $y_i$ is the true class index for sample $i$.

```python
import math

def cross_entropy_loss(predictions: list[list[float]], 
                       targets: list[int]) -> float:
    """
    Cross-entropy loss for multi-class classification
    predictions: (batch_size, num_classes) - logits or probabilities
    targets: (batch_size,) - class indices
    """
    batch_size = len(predictions)
    
    # Apply softmax to get probabilities
    probs = []
    for pred in predictions:
        max_p = max(pred)
        exp_p = [math.exp(p - max_p) for p in pred]
        sum_exp = sum(exp_p)
        probs.append([e / sum_exp for e in exp_p])
    
    # Compute negative log likelihood
    loss = 0.0
    for i in range(batch_size):
        loss -= math.log(probs[i][targets[i]] + 1e-10)
    
    return loss / batch_size

# Time: O(batch * classes), Space: O(batch * classes)

# Examples:
# Input: predictions = [[2.0, 1.0, 0.1]], targets = [0]
# Output: 0.417  (model correctly predicts class 0)

# Input: predictions = [[0.1, 0.1, 2.0]], targets = [0]
# Output: 2.31  (model incorrectly predicts class 2)
```

---

### Problem 22: Implement Binary Cross-Entropy Loss
**Difficulty:** Easy

Implement BCE loss for binary classification.

**Formula:**
$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \left[y_i \log(p_i) + (1-y_i)\log(1-p_i)\right]$$

**Numerically stable (from logits z):**
$$\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N} \left[\max(z_i, 0) - z_i \cdot y_i + \log(1 + e^{-|z_i|})\right]$$

```python
import math

def binary_cross_entropy(predictions: list[float], targets: list[float]) -> float:
    """
    BCE = -1/N * sum(y*log(p) + (1-y)*log(1-p))
    predictions: probabilities in [0, 1]
    targets: binary labels (0 or 1)
    """
    n = len(predictions)
    loss = 0.0
    eps = 1e-10
    
    for p, y in zip(predictions, targets):
        p = max(min(p, 1 - eps), eps)  # Clip for numerical stability
        loss -= y * math.log(p) + (1 - y) * math.log(1 - p)
    
    return loss / n

def bce_with_logits(logits: list[float], targets: list[float]) -> float:
    """
    Numerically stable BCE from logits
    """
    n = len(logits)
    loss = 0.0
    
    for z, y in zip(logits, targets):
        # max(z, 0) - z*y + log(1 + exp(-|z|))
        loss += max(z, 0) - z * y + math.log(1 + math.exp(-abs(z)))
    
    return loss / n

# Examples:
# Input: predictions = [0.9, 0.1], targets = [1, 0]
# Output: 0.105  (good predictions)

# Input: predictions = [0.1, 0.9], targets = [1, 0]
# Output: 2.30  (bad predictions)
```

---

### Problem 23: Implement MSE Loss with Gradient
**Difficulty:** Easy

Implement MSE loss and its gradient.

**Formula:**
$$\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

**Gradient:**
$$\frac{\partial \mathcal{L}}{\partial \hat{y}_i} = \frac{2}{N}(\hat{y}_i - y_i)$$

```python
def mse_loss(predictions: list[float], targets: list[float]) -> float:
    """Mean Squared Error: (1/n) * sum((pred - target)^2)"""
    n = len(predictions)
    return sum((p - t) ** 2 for p, t in zip(predictions, targets)) / n

def mse_gradient(predictions: list[float], targets: list[float]) -> list[float]:
    """Gradient of MSE: (2/n) * (pred - target)"""
    n = len(predictions)
    return [(2 / n) * (p - t) for p, t in zip(predictions, targets)]

# Examples:
# Input: predictions = [1, 2, 3], targets = [1, 2, 3]
# Output: loss = 0.0, gradient = [0, 0, 0]

# Input: predictions = [2, 3, 4], targets = [1, 2, 3]
# Output: loss = 1.0, gradient = [0.67, 0.67, 0.67]
```

---

### Problem 24: Implement Accuracy, Precision, Recall, F1
**Difficulty:** Medium

Implement classification metrics from scratch.

**Formulas:**
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{Precision} = \frac{TP}{TP + FP}$$

$$\text{Recall} = \frac{TP}{TP + FN}$$

$$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}$$

```python
def confusion_matrix(y_true: list[int], y_pred: list[int]) -> dict:
    """Compute TP, TN, FP, FN for binary classification"""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}

def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    """Accuracy = (TP + TN) / Total"""
    cm = confusion_matrix(y_true, y_pred)
    total = cm['tp'] + cm['tn'] + cm['fp'] + cm['fn']
    return (cm['tp'] + cm['tn']) / total if total > 0 else 0

def precision(y_true: list[int], y_pred: list[int]) -> float:
    """Precision = TP / (TP + FP)"""
    cm = confusion_matrix(y_true, y_pred)
    denom = cm['tp'] + cm['fp']
    return cm['tp'] / denom if denom > 0 else 0

def recall(y_true: list[int], y_pred: list[int]) -> float:
    """Recall = TP / (TP + FN)"""
    cm = confusion_matrix(y_true, y_pred)
    denom = cm['tp'] + cm['fn']
    return cm['tp'] / denom if denom > 0 else 0

def f1_score(y_true: list[int], y_pred: list[int]) -> float:
    """F1 = 2 * (Precision * Recall) / (Precision + Recall)"""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0

# Examples:
# Input: y_true = [1,1,0,0,1], y_pred = [1,0,0,1,1]
# Output: accuracy=0.6, precision=0.67, recall=0.67, f1=0.67
```

---

### Problem 25: Implement IoU (Intersection over Union)
**Difficulty:** Medium

Implement IoU for bounding boxes (object detection).

**Formula:**
$$\text{IoU} = \frac{|A \cap B|}{|A \cup B|} = \frac{\text{Area of Intersection}}{\text{Area of Union}}$$

$$\text{IoU} = \frac{\text{Intersection}}{\text{Area}_A + \text{Area}_B - \text{Intersection}}$$

Range: [0, 1] where 1 = perfect overlap, 0 = no overlap

```python
def iou(box1: list[float], box2: list[float]) -> float:
    """
    Compute IoU of two bounding boxes
    box format: [x1, y1, x2, y2] (top-left and bottom-right corners)
    """
    # Intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Intersection area
    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    intersection = inter_width * inter_height
    
    # Union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

# Time: O(1), Space: O(1)

# Examples:
# Input: box1 = [0,0,2,2], box2 = [1,1,3,3]
# Output: 0.143  (1/7, small overlap)

# Input: box1 = [0,0,2,2], box2 = [0,0,2,2]
# Output: 1.0  (perfect overlap)

# Input: box1 = [0,0,1,1], box2 = [2,2,3,3]
# Output: 0.0  (no overlap)
```

---

### Problem 26: Implement Non-Maximum Suppression (NMS)
**Difficulty:** Hard

Implement NMS for object detection post-processing.

```python
def nms(boxes: list[list[float]], scores: list[float], 
        iou_threshold: float = 0.5) -> list[int]:
    """
    Non-Maximum Suppression
    boxes: list of [x1, y1, x2, y2]
    scores: confidence scores for each box
    Returns: indices of boxes to keep
    """
    if not boxes:
        return []
    
    # Sort by score (descending)
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    
    while indices:
        # Keep the highest scoring box
        current = indices[0]
        keep.append(current)
        indices = indices[1:]
        
        # Remove boxes with high IoU overlap
        remaining = []
        for idx in indices:
            if iou(boxes[current], boxes[idx]) < iou_threshold:
                remaining.append(idx)
        indices = remaining
    
    return keep

# Time: O(n²), Space: O(n)

# Examples:
# Input: boxes = [[0,0,2,2], [0.1,0.1,2.1,2.1], [3,3,5,5]]
#        scores = [0.9, 0.8, 0.7], iou_threshold = 0.5
# Output: [0, 2]  (box 1 suppressed due to high IoU with box 0)
```

---

### Problem 27: Implement Top-K Accuracy
**Difficulty:** Easy

Implement top-k accuracy metric.

```python
def top_k_accuracy(predictions: list[list[float]], targets: list[int], 
                   k: int = 5) -> float:
    """
    Top-k accuracy: correct if true label is in top k predictions
    predictions: (batch_size, num_classes) - scores/logits
    targets: (batch_size,) - true class indices
    """
    correct = 0
    
    for pred, target in zip(predictions, targets):
        # Get indices of top k predictions
        top_k_indices = sorted(range(len(pred)), key=lambda i: pred[i], 
                               reverse=True)[:k]
        if target in top_k_indices:
            correct += 1
    
    return correct / len(targets)

# Examples:
# Input: predictions = [[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]], targets = [2, 0], k=1
# Output: 1.0  (both correct in top-1)

# Input: predictions = [[0.1, 0.2, 0.7], [0.1, 0.8, 0.1]], targets = [0, 0], k=2
# Output: 0.5  (first wrong in top-2, second correct)
```

---

### Problem 28: Implement AUC-ROC
**Difficulty:** Hard

Implement Area Under ROC Curve.

```python
def auc_roc(y_true: list[int], y_scores: list[float]) -> float:
    """
    Compute AUC-ROC using the trapezoidal rule
    """
    # Sort by scores descending
    pairs = sorted(zip(y_scores, y_true), reverse=True)
    
    # Count positives and negatives
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return 0.5
    
    # Compute TPR and FPR at each threshold
    tpr_list, fpr_list = [0], [0]
    tp, fp = 0, 0
    
    for score, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)
    
    # Compute AUC using trapezoidal rule
    auc = 0
    for i in range(1, len(fpr_list)):
        auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2
    
    return auc

# Examples:
# Input: y_true = [1,1,0,0], y_scores = [0.9, 0.8, 0.3, 0.2]
# Output: 1.0  (perfect separation)

# Input: y_true = [1,0,1,0], y_scores = [0.5, 0.5, 0.5, 0.5]
# Output: 0.5  (random classifier)
```

---

## Data Processing & Augmentation

### Problem 29: Implement One-Hot Encoding
**Difficulty:** Easy

Implement one-hot encoding for categorical labels.

```python
def one_hot_encode(labels: list[int], num_classes: int) -> list[list[int]]:
    """
    Convert integer labels to one-hot vectors
    """
    result = []
    for label in labels:
        one_hot = [0] * num_classes
        one_hot[label] = 1
        result.append(one_hot)
    return result

def one_hot_decode(one_hot: list[list[int]]) -> list[int]:
    """Convert one-hot vectors back to integer labels"""
    return [row.index(max(row)) for row in one_hot]

# Examples:
# Input: labels = [0, 2, 1], num_classes = 3
# Output: [[1,0,0], [0,0,1], [0,1,0]]

# Input: labels = [3], num_classes = 5
# Output: [[0,0,0,1,0]]
```

---

### Problem 30: Implement Min-Max Normalization
**Difficulty:** Easy

Implement min-max scaling to [0, 1] range.

**Formula:**
$$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$$

Output range: [0, 1]

```python
def min_max_normalize(data: list[list[float]]) -> tuple:
    """
    Normalize each feature to [0, 1] range
    Returns: (normalized_data, min_vals, max_vals)
    """
    n_samples, n_features = len(data), len(data[0])
    
    # Compute min and max for each feature
    min_vals = [min(data[i][j] for i in range(n_samples)) for j in range(n_features)]
    max_vals = [max(data[i][j] for i in range(n_samples)) for j in range(n_features)]
    
    # Normalize
    normalized = []
    for i in range(n_samples):
        row = []
        for j in range(n_features):
            range_val = max_vals[j] - min_vals[j]
            if range_val > 0:
                row.append((data[i][j] - min_vals[j]) / range_val)
            else:
                row.append(0.0)
        normalized.append(row)
    
    return normalized, min_vals, max_vals

# Examples:
# Input: data = [[1, 10], [2, 20], [3, 30]]
# Output: [[0, 0], [0.5, 0.5], [1, 1]]
```

---

### Problem 31: Implement Z-Score Standardization
**Difficulty:** Easy

Implement standardization (mean=0, std=1).

**Formula:**
$$x' = \frac{x - \mu}{\sigma}$$

Where $\mu$ is mean and $\sigma$ is standard deviation.
Output: mean=0, std=1

```python
import math

def standardize(data: list[list[float]]) -> tuple:
    """
    Standardize each feature to mean=0, std=1
    Returns: (standardized_data, means, stds)
    """
    n_samples, n_features = len(data), len(data[0])
    
    # Compute mean for each feature
    means = [sum(data[i][j] for i in range(n_samples)) / n_samples 
             for j in range(n_features)]
    
    # Compute std for each feature
    stds = []
    for j in range(n_features):
        variance = sum((data[i][j] - means[j])**2 for i in range(n_samples)) / n_samples
        stds.append(math.sqrt(variance) if variance > 0 else 1.0)
    
    # Standardize
    standardized = [[(data[i][j] - means[j]) / stds[j] 
                     for j in range(n_features)] for i in range(n_samples)]
    
    return standardized, means, stds

# Examples:
# Input: data = [[1, 100], [2, 200], [3, 300]]
# Output: [[-1.22, -1.22], [0, 0], [1.22, 1.22]]
```

---

### Problem 32: Implement Data Augmentation - Random Crop
**Difficulty:** Medium

Implement random crop for image augmentation.

```python
import random

def random_crop(image: list[list[float]], crop_h: int, crop_w: int) -> list[list[float]]:
    """
    Randomly crop a region from the image
    image: (H, W)
    """
    H, W = len(image), len(image[0])
    
    assert crop_h <= H and crop_w <= W, "Crop size larger than image"
    
    # Random top-left corner
    top = random.randint(0, H - crop_h)
    left = random.randint(0, W - crop_w)
    
    # Extract crop
    return [image[i][left:left + crop_w] for i in range(top, top + crop_h)]

def center_crop(image: list[list[float]], crop_h: int, crop_w: int) -> list[list[float]]:
    """Center crop"""
    H, W = len(image), len(image[0])
    top = (H - crop_h) // 2
    left = (W - crop_w) // 2
    return [image[i][left:left + crop_w] for i in range(top, top + crop_h)]

# Examples:
# Input: image = 5x5 matrix, crop_h=3, crop_w=3
# Output: Random 3x3 region from the image
```

---

### Problem 33: Implement Horizontal Flip
**Difficulty:** Easy

Implement horizontal flip augmentation.

```python
def horizontal_flip(image: list[list[float]]) -> list[list[float]]:
    """Flip image horizontally (left-right)"""
    return [row[::-1] for row in image]

def vertical_flip(image: list[list[float]]) -> list[list[float]]:
    """Flip image vertically (top-bottom)"""
    return image[::-1]

def random_horizontal_flip(image: list[list[float]], p: float = 0.5) -> list[list[float]]:
    """Randomly flip with probability p"""
    import random
    if random.random() < p:
        return horizontal_flip(image)
    return image

# Examples:
# Input: image = [[1,2,3],[4,5,6]]
# horizontal_flip output: [[3,2,1],[6,5,4]]
# vertical_flip output: [[4,5,6],[1,2,3]]
```

---

### Problem 34: Implement Label Smoothing
**Difficulty:** Easy

Implement label smoothing for regularization.

**Formula:**
$$y'_i = \begin{cases} 1 - \epsilon + \frac{\epsilon}{K} & \text{if } i = \text{true class} \\ \frac{\epsilon}{K} & \text{otherwise} \end{cases}$$

Where $\epsilon$ is smoothing factor (e.g., 0.1) and $K$ is number of classes.

Example: Hard [0,0,1,0] → Soft [0.025, 0.025, 0.925, 0.025] for $\epsilon=0.1$, $K=4$

```python
def label_smoothing(labels: list[int], num_classes: int, 
                    smoothing: float = 0.1) -> list[list[float]]:
    """
    Convert hard labels to soft labels with smoothing
    Hard: [0, 0, 1, 0] → Soft: [0.025, 0.025, 0.925, 0.025] (for smoothing=0.1)
    """
    result = []
    smooth_value = smoothing / num_classes
    confident_value = 1.0 - smoothing + smooth_value
    
    for label in labels:
        soft_label = [smooth_value] * num_classes
        soft_label[label] = confident_value
        result.append(soft_label)
    
    return result

# Examples:
# Input: labels = [2], num_classes = 4, smoothing = 0.1
# Output: [[0.025, 0.025, 0.925, 0.025]]

# Input: labels = [0, 1], num_classes = 3, smoothing = 0.2
# Output: [[0.867, 0.067, 0.067], [0.067, 0.867, 0.067]]
```

---

### Problem 35: Implement Mixup Data Augmentation
**Difficulty:** Medium

Implement mixup augmentation for training.

**Formula:**
$$\tilde{x} = \lambda x_i + (1-\lambda) x_j$$
$$\tilde{y} = \lambda y_i + (1-\lambda) y_j$$

Where $\lambda \sim \text{Beta}(\alpha, \alpha)$ and $(x_i, y_i)$, $(x_j, y_j)$ are random training pairs.

```python
import random

def mixup(x1: list[float], y1: list[float], x2: list[float], y2: list[float],
          alpha: float = 0.2) -> tuple:
    """
    Mixup: create virtual training examples
    x_mixed = lambda * x1 + (1 - lambda) * x2
    y_mixed = lambda * y1 + (1 - lambda) * y2
    lambda ~ Beta(alpha, alpha)
    """
    # Sample lambda from Beta distribution (simplified: uniform for demo)
    lam = random.betavariate(alpha, alpha) if alpha > 0 else 0.5
    
    x_mixed = [lam * a + (1 - lam) * b for a, b in zip(x1, x2)]
    y_mixed = [lam * a + (1 - lam) * b for a, b in zip(y1, y2)]
    
    return x_mixed, y_mixed, lam

# Examples:
# Input: x1 = [1,0], y1 = [1,0,0], x2 = [0,1], y2 = [0,1,0], lam = 0.7
# Output: x_mixed = [0.7, 0.3], y_mixed = [0.7, 0.3, 0]
```

---

### Problem 36: Implement Train/Val/Test Split
**Difficulty:** Easy

Implement data splitting with shuffling.

```python
import random

def train_val_test_split(data: list, labels: list, 
                         train_ratio: float = 0.7,
                         val_ratio: float = 0.15,
                         seed: int = 42) -> tuple:
    """
    Split data into train, validation, and test sets
    """
    random.seed(seed)
    n = len(data)
    indices = list(range(n))
    random.shuffle(indices)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    X_train = [data[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]
    X_val = [data[i] for i in val_idx]
    y_val = [labels[i] for i in val_idx]
    X_test = [data[i] for i in test_idx]
    y_test = [labels[i] for i in test_idx]
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# Examples:
# Input: data with 100 samples, train=0.7, val=0.15, test=0.15
# Output: 70 train, 15 val, 15 test samples
```

---

## Optimization & Training


### Problem 37: Implement Learning Rate Scheduler
**Difficulty:** Medium

Implement common learning rate schedules.

**Step Decay:**
$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t / s \rfloor}$$

**Exponential Decay:**
$$\eta_t = \eta_0 \cdot \gamma^t$$

**Cosine Annealing:**
$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$$

**Linear Warmup:**
$$\eta_t = \eta_{target} \cdot \frac{t}{T_{warmup}}$$ for $t < T_{warmup}$

```python
import math

def step_lr(initial_lr: float, epoch: int, step_size: int = 10, 
            gamma: float = 0.1) -> float:
    """Step decay: lr = initial_lr * gamma^(epoch // step_size)"""
    return initial_lr * (gamma ** (epoch // step_size))

def exponential_lr(initial_lr: float, epoch: int, gamma: float = 0.95) -> float:
    """Exponential decay: lr = initial_lr * gamma^epoch"""
    return initial_lr * (gamma ** epoch)

def cosine_annealing_lr(initial_lr: float, epoch: int, total_epochs: int,
                        min_lr: float = 0) -> float:
    """Cosine annealing: smooth decay to min_lr"""
    return min_lr + (initial_lr - min_lr) * (1 + math.cos(math.pi * epoch / total_epochs)) / 2

def warmup_lr(initial_lr: float, epoch: int, warmup_epochs: int = 5) -> float:
    """Linear warmup"""
    if epoch < warmup_epochs:
        return initial_lr * (epoch + 1) / warmup_epochs
    return initial_lr

# Examples:
# step_lr(0.1, epoch=25, step_size=10, gamma=0.1) → 0.001
# cosine_annealing_lr(0.1, epoch=50, total_epochs=100) → 0.05
# warmup_lr(0.1, epoch=2, warmup_epochs=5) → 0.06
```

---

### Problem 38: Implement Early Stopping
**Difficulty:** Medium

Implement early stopping logic for training.

```python
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        """
        patience: epochs to wait before stopping
        min_delta: minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
    
    def step(self, val_loss: float) -> bool:
        """
        Call after each epoch with validation loss
        Returns True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop

# Example usage:
# early_stop = EarlyStopping(patience=3)
# for epoch in range(100):
#     train_loss = train_one_epoch()
#     val_loss = validate()
#     if early_stop.step(val_loss):
#         print(f"Early stopping at epoch {epoch}")
#         break
```

---

### Problem 39: Implement Gradient Clipping
**Difficulty:** Easy

Implement gradient clipping by norm.

**Clip by Global Norm:**
$$\|g\| = \sqrt{\sum_i g_i^2}$$
$$g' = g \cdot \frac{\text{max\_norm}}{\max(\|g\|, \text{max\_norm})}$$

**Clip by Value:**
$$g'_i = \text{clip}(g_i, -c, c)$$

```python
import math

def clip_grad_norm(gradients: list[list[float]], max_norm: float) -> list[list[float]]:
    """
    Clip gradients so that total norm <= max_norm
    """
    # Compute total norm
    total_norm = 0.0
    for grad in gradients:
        for g in grad:
            total_norm += g ** 2
    total_norm = math.sqrt(total_norm)
    
    # Clip if necessary
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [[g * scale for g in grad] for grad in gradients]
    
    return gradients

def clip_grad_value(gradients: list[list[float]], clip_value: float) -> list[list[float]]:
    """Clip each gradient element to [-clip_value, clip_value]"""
    return [[max(-clip_value, min(clip_value, g)) for g in grad] for grad in gradients]

# Examples:
# Input: gradients = [[3, 4]], max_norm = 1.0
# Total norm = 5, scale = 0.2
# Output: [[0.6, 0.8]]  (norm = 1.0)
```

---

### Problem 40: Implement Mini-Batch Generator
**Difficulty:** Easy

Implement a mini-batch data generator.

```python
import random

def batch_generator(X: list, y: list, batch_size: int, shuffle: bool = True):
    """
    Generate mini-batches for training
    """
    n = len(X)
    indices = list(range(n))
    
    if shuffle:
        random.shuffle(indices)
    
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_indices = indices[start:end]
        
        X_batch = [X[i] for i in batch_indices]
        y_batch = [y[i] for i in batch_indices]
        
        yield X_batch, y_batch

# Example usage:
# for X_batch, y_batch in batch_generator(X_train, y_train, batch_size=32):
#     loss = train_step(X_batch, y_batch)
```

---

### Problem 41: Implement K-Fold Cross Validation
**Difficulty:** Medium

Implement k-fold cross validation splitting.

```python
def k_fold_split(data: list, k: int = 5) -> list:
    """
    Split data into k folds
    Returns: list of (train_indices, val_indices) tuples
    """
    n = len(data)
    fold_size = n // k
    indices = list(range(n))
    
    folds = []
    for i in range(k):
        val_start = i * fold_size
        val_end = val_start + fold_size if i < k - 1 else n
        
        val_indices = indices[val_start:val_end]
        train_indices = indices[:val_start] + indices[val_end:]
        
        folds.append((train_indices, val_indices))
    
    return folds

# Example:
# Input: data with 100 samples, k=5
# Output: 5 folds, each with 80 train and 20 val samples
# Fold 0: train=[20-99], val=[0-19]
# Fold 1: train=[0-19,40-99], val=[20-39]
# ...
```

---

### Problem 42: Implement Exponential Moving Average
**Difficulty:** Easy

Implement EMA for model weights (used in training).

**Formula:**
$$\theta_{EMA}^{(t)} = \alpha \cdot \theta_{EMA}^{(t-1)} + (1-\alpha) \cdot \theta^{(t)}$$

Where $\alpha$ is decay rate (typically 0.999). Higher $\alpha$ = smoother, slower updates.

```python
def ema_update(ema_weights: list[float], model_weights: list[float], 
               decay: float = 0.999) -> list[float]:
    """
    Update EMA weights: ema = decay * ema + (1 - decay) * model
    """
    return [decay * e + (1 - decay) * m 
            for e, m in zip(ema_weights, model_weights)]

class EMA:
    def __init__(self, decay: float = 0.999):
        self.decay = decay
        self.shadow = None
    
    def update(self, model_weights: list[float]):
        if self.shadow is None:
            self.shadow = model_weights.copy()
        else:
            self.shadow = ema_update(self.shadow, model_weights, self.decay)
    
    def get_weights(self) -> list[float]:
        return self.shadow

# Example:
# EMA smooths model weights over training for more stable predictions
# decay=0.999 means 99.9% old weights + 0.1% new weights
```

---

## Computer Vision

### Problem 43: Implement Image Resize (Nearest Neighbor)
**Difficulty:** Medium

Implement image resizing using nearest neighbor interpolation.

```python
def resize_nearest(image: list[list[float]], new_h: int, new_w: int) -> list[list[float]]:
    """
    Resize image using nearest neighbor interpolation
    """
    old_h, old_w = len(image), len(image[0])
    
    result = [[0.0] * new_w for _ in range(new_h)]
    
    for i in range(new_h):
        for j in range(new_w):
            # Map new coordinates to old coordinates
            old_i = int(i * old_h / new_h)
            old_j = int(j * old_w / new_w)
            
            # Clamp to valid range
            old_i = min(old_i, old_h - 1)
            old_j = min(old_j, old_w - 1)
            
            result[i][j] = image[old_i][old_j]
    
    return result

# Examples:
# Input: 2x2 image [[1,2],[3,4]], new_h=4, new_w=4
# Output: 4x4 image with each pixel duplicated
# [[1,1,2,2],[1,1,2,2],[3,3,4,4],[3,3,4,4]]
```

---

### Problem 44: Implement Bilinear Interpolation
**Difficulty:** Hard

Implement bilinear interpolation for image resizing.

```python
def resize_bilinear(image: list[list[float]], new_h: int, new_w: int) -> list[list[float]]:
    """
    Resize image using bilinear interpolation
    """
    old_h, old_w = len(image), len(image[0])
    result = [[0.0] * new_w for _ in range(new_h)]
    
    for i in range(new_h):
        for j in range(new_w):
            # Map to old coordinates (with fractional part)
            old_i = i * (old_h - 1) / (new_h - 1) if new_h > 1 else 0
            old_j = j * (old_w - 1) / (new_w - 1) if new_w > 1 else 0
            
            # Get integer and fractional parts
            i0, j0 = int(old_i), int(old_j)
            i1, j1 = min(i0 + 1, old_h - 1), min(j0 + 1, old_w - 1)
            di, dj = old_i - i0, old_j - j0
            
            # Bilinear interpolation
            result[i][j] = (
                image[i0][j0] * (1 - di) * (1 - dj) +
                image[i0][j1] * (1 - di) * dj +
                image[i1][j0] * di * (1 - dj) +
                image[i1][j1] * di * dj
            )
    
    return result

# Bilinear produces smoother results than nearest neighbor
```

---

### Problem 45: Implement Sobel Edge Detection
**Difficulty:** Medium

Implement Sobel operator for edge detection.

```python
def sobel_edge_detection(image: list[list[float]]) -> list[list[float]]:
    """
    Apply Sobel operator for edge detection
    """
    # Sobel kernels
    sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    
    H, W = len(image), len(image[0])
    
    # Apply convolution (simplified, no padding)
    result = [[0.0] * (W - 2) for _ in range(H - 2)]
    
    for i in range(H - 2):
        for j in range(W - 2):
            gx, gy = 0.0, 0.0
            for ki in range(3):
                for kj in range(3):
                    pixel = image[i + ki][j + kj]
                    gx += pixel * sobel_x[ki][kj]
                    gy += pixel * sobel_y[ki][kj]
            
            # Gradient magnitude
            result[i][j] = (gx**2 + gy**2) ** 0.5
    
    return result

# Example: Detects edges (high gradient regions) in images
```

---

### Problem 46: Implement Gaussian Blur
**Difficulty:** Medium

Implement Gaussian blur for image smoothing.

**Gaussian Kernel Formula:**
$$G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}$$

**1D Gaussian (separable):**
$$G(x) = \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{x^2}{2\sigma^2}}$$

```python
import math

def gaussian_kernel(size: int, sigma: float) -> list[list[float]]:
    """Generate a Gaussian kernel"""
    kernel = [[0.0] * size for _ in range(size)]
    center = size // 2
    total = 0.0
    
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i][j] = math.exp(-(x**2 + y**2) / (2 * sigma**2))
            total += kernel[i][j]
    
    # Normalize
    for i in range(size):
        for j in range(size):
            kernel[i][j] /= total
    
    return kernel

def gaussian_blur(image: list[list[float]], kernel_size: int = 3, 
                  sigma: float = 1.0) -> list[list[float]]:
    """Apply Gaussian blur to image"""
    kernel = gaussian_kernel(kernel_size, sigma)
    return conv2d(image, kernel)  # Use conv2d from Problem 4

# Example: Smooths image by averaging with Gaussian-weighted neighbors
```

---

### Problem 47: Implement RGB to Grayscale
**Difficulty:** Easy

Convert RGB image to grayscale.

**Formula (ITU-R BT.601):**
$$Y = 0.299R + 0.587G + 0.114B$$

The weights reflect human eye sensitivity (most sensitive to green, least to blue).

```python
def rgb_to_grayscale(image: list[list[list[float]]]) -> list[list[float]]:
    """
    Convert RGB image to grayscale
    image: (H, W, 3) - RGB channels
    output: (H, W) - grayscale
    Formula: Y = 0.299*R + 0.587*G + 0.114*B
    """
    H, W = len(image), len(image[0])
    
    grayscale = [[0.0] * W for _ in range(H)]
    
    for i in range(H):
        for j in range(W):
            r, g, b = image[i][j]
            grayscale[i][j] = 0.299 * r + 0.587 * g + 0.114 * b
    
    return grayscale

# Examples:
# Input: [[[255, 0, 0]]]  (pure red)
# Output: [[76.245]]

# Input: [[[0, 255, 0]]]  (pure green)
# Output: [[149.685]]

# Input: [[[255, 255, 255]]]  (white)
# Output: [[255.0]]
```

---

## Sequence & Time Series

### Problem 48: Implement Positional Encoding
**Difficulty:** Medium

Implement sinusoidal positional encoding for transformers.

**Formula:**
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

Where $pos$ is position and $i$ is dimension index.

```python
import math

def positional_encoding(seq_len: int, d_model: int) -> list[list[float]]:
    """
    Generate sinusoidal positional encodings
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    pe = [[0.0] * d_model for _ in range(seq_len)]
    
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            div_term = 10000 ** (i / d_model)
            pe[pos][i] = math.sin(pos / div_term)
            if i + 1 < d_model:
                pe[pos][i + 1] = math.cos(pos / div_term)
    
    return pe

# Examples:
# Input: seq_len=4, d_model=4
# Output: 4x4 matrix of positional encodings
# Position 0: [0, 1, 0, 1]  (sin(0)=0, cos(0)=1)
# Position 1: [0.84, 0.54, 0.01, 1.0]  (approximately)
```

---

### Problem 49: Implement Causal Mask for Attention
**Difficulty:** Easy

Implement causal (look-ahead) mask for autoregressive models.

```python
def causal_mask(seq_len: int) -> list[list[float]]:
    """
    Create causal mask: positions can only attend to previous positions
    Returns: (seq_len, seq_len) mask where mask[i][j] = 0 if j > i else 1
    """
    mask = [[0.0] * seq_len for _ in range(seq_len)]
    
    for i in range(seq_len):
        for j in range(seq_len):
            if j <= i:
                mask[i][j] = 1.0
            else:
                mask[i][j] = 0.0  # or float('-inf') for additive mask
    
    return mask

def apply_causal_mask(attention_scores: list[list[float]]) -> list[list[float]]:
    """Apply causal mask to attention scores (set future to -inf)"""
    seq_len = len(attention_scores)
    masked = [[0.0] * seq_len for _ in range(seq_len)]
    
    for i in range(seq_len):
        for j in range(seq_len):
            if j <= i:
                masked[i][j] = attention_scores[i][j]
            else:
                masked[i][j] = float('-inf')
    
    return masked

# Examples:
# Input: seq_len = 4
# Output: [[1,0,0,0],
#          [1,1,0,0],
#          [1,1,1,0],
#          [1,1,1,1]]
```

---

### Problem 50: Implement Sliding Window for Time Series
**Difficulty:** Easy

Create sliding windows for time series prediction.

```python
def create_sequences(data: list[float], seq_length: int, 
                     pred_length: int = 1) -> tuple:
    """
    Create input-output pairs for time series prediction
    data: time series values
    seq_length: number of past values to use as input
    pred_length: number of future values to predict
    """
    X, y = [], []
    
    for i in range(len(data) - seq_length - pred_length + 1):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length:i + seq_length + pred_length])
    
    return X, y

def create_sequences_multivariate(data: list[list[float]], seq_length: int,
                                   target_col: int = 0) -> tuple:
    """
    Create sequences for multivariate time series
    data: (timesteps, features)
    """
    X, y = [], []
    
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length][target_col])
    
    return X, y

# Examples:
# Input: data = [1,2,3,4,5,6,7,8,9,10], seq_length=3, pred_length=1
# Output: X = [[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9]]
#         y = [[4],[5],[6],[7],[8],[9],[10]]
```

---

## Bonus: ML Algorithm Implementations

### Bonus 1: Implement K-Means Clustering
**Difficulty:** Medium

**Algorithm:**
1. Initialize $k$ centroids randomly
2. Assign each point to nearest centroid: $c_i = \arg\min_j \|x_i - \mu_j\|^2$
3. Update centroids: $\mu_j = \frac{1}{|C_j|}\sum_{x_i \in C_j} x_i$
4. Repeat until convergence

**Objective (minimize):**
$$J = \sum_{j=1}^{k}\sum_{x_i \in C_j} \|x_i - \mu_j\|^2$$

```python
import random
import math

def kmeans(data: list[list[float]], k: int, max_iters: int = 100) -> tuple:
    """
    K-Means clustering
    Returns: (centroids, labels)
    """
    n, d = len(data), len(data[0])
    
    # Initialize centroids randomly
    centroids = random.sample(data, k)
    
    for _ in range(max_iters):
        # Assign points to nearest centroid
        labels = []
        for point in data:
            distances = [sum((point[i] - c[i])**2 for i in range(d)) 
                        for c in centroids]
            labels.append(distances.index(min(distances)))
        
        # Update centroids
        new_centroids = []
        for j in range(k):
            cluster_points = [data[i] for i in range(n) if labels[i] == j]
            if cluster_points:
                new_centroid = [sum(p[i] for p in cluster_points) / len(cluster_points)
                               for i in range(d)]
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(centroids[j])
        
        # Check convergence
        if new_centroids == centroids:
            break
        centroids = new_centroids
    
    return centroids, labels

# Example: Clusters data points into k groups
```

---

### Bonus 2: Implement KNN Classifier
**Difficulty:** Easy

**Distance (Euclidean):**
$$d(x, x') = \sqrt{\sum_{i=1}^{n}(x_i - x'_i)^2}$$

**Prediction:**
$$\hat{y} = \text{mode}(\{y_i : x_i \in N_k(x)\})$$

Where $N_k(x)$ is the set of $k$ nearest neighbors of $x$.

```python
from collections import Counter
import math

def knn_predict(X_train: list[list[float]], y_train: list[int],
                x_test: list[float], k: int = 3) -> int:
    """
    K-Nearest Neighbors prediction for a single point
    """
    # Compute distances to all training points
    distances = []
    for i, x in enumerate(X_train):
        dist = math.sqrt(sum((a - b)**2 for a, b in zip(x, x_test)))
        distances.append((dist, y_train[i]))
    
    # Sort by distance and get k nearest
    distances.sort(key=lambda x: x[0])
    k_nearest = [label for _, label in distances[:k]]
    
    # Majority vote
    return Counter(k_nearest).most_common(1)[0][0]

# Example:
# Input: X_train = [[0,0],[1,1],[2,2]], y_train = [0,0,1], x_test = [1.5,1.5], k=2
# Output: 0 or 1 depending on nearest neighbors
```

---

## Quick Reference: Common ML Operations Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Matrix Multiply (m×n, n×p) | O(mnp) | O(mp) |
| Softmax | O(n) | O(n) |
| Convolution 2D | O(H×W×kH×kW) | O(H×W) |
| Attention | O(n²d) | O(n²) |
| Batch Norm | O(batch×features) | O(features) |
| Dropout | O(n) | O(n) |
| NMS | O(n²) | O(n) |
| K-Means (1 iter) | O(nkd) | O(k×d) |

---
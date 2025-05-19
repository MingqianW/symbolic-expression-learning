## 📘 Symbolic Regression: Architectural Progress & Observations

### 🔍 True Function Example 1

```
y = 0.8 * (0.3 * x1 + 1.4 * x2 + 0.7 * x3 + 0.6)^0.5
```

#### 🔹 Mixed Model (Shallow Architecture)

```
Epoch 11200, Loss: 0.000005  Test Loss: 0.000005
Learned symbolic expression:
0.745 * (x1^-0.014 * x2^0.153 * x3^-0.001) + (0.066*x1 + 0.192*x2 + 0.138*x3 + 0.257)
```

- ❗ Despite low loss, the symbolic form is **not aligned** with the true structure.
- **Interpretation**: The model fails to discover the inner linear form due to limited depth.

#### 🔹 Sparse Pruned Model (Deeper Compositional Structure)

```
Simplified symbolic expression:
y = 0.892 * ((0.210*x1 + 0.981*x2 + 0.490*x3 + 0.610))^0.533  (i.e., y3)
```

- ✅ Much closer to ground truth — **captures the linear transform and exponentiation**.
- **Conclusion**: Making the network **deeper and modular** helps uncover latent structure.

---

### 🔍 True Function Example 2

```
y = 0.4 * (x1^0.3 * x2^0.7 * x3^0.6) + 0.3
```

#### 🔹 Sparse Pruned Model

```
Simplified symbolic expression:
y = 0.386 * (x1^0.306 * x2^0.715 * x3^0.613) + 0.316  (i.e., y4)
```

- ✅ Precisely identifies **product-based structure** and accurate exponents.
- **Conclusion**: The model generalizes well across different structural forms.

---

### 🧠 Expression Form Learned by Sparse Model (Symbolically)

We assume:

- First-layer outputs:
  - Power: y₁ = C₁ * ∏ xᵢ^wᵢ
  - Linear: y₂ = ∑ aᵢ * xᵢ + b₁
- Second-layer operations:
  - Power activation: y₃ = C₂ * y₁^w₂
  - Linear transformation: y₄ = a₂ * y₂ + b₂

Combined:

```
y = C₂ * (C₁ * ∏ xᵢ^wᵢ)^w₂ + a₂ * (∑ aᵢ * xᵢ + b₁) + b₂
```

Simplified:

```
y = [C₂ * C₁^w₂] * ∏ xᵢ^(wᵢ * w₂) + a₂ * ∑ aᵢ * xᵢ + a₂ * b₁ + b₂
```

---

### ⚠️ Challenges & Insights

- **Two Key Tasks**:
  1. Identify the structure of the function (e.g., product vs. sum).
  2. Learn accurate coefficients and exponents.

- **Parameter Initialization Matters**:
  - If the true parameter is outside the initialized range, it becomes hard to reach.
  - Gradient-based learning is sensitive to **parameter scale imbalance**.
  - Especially critical for:
    - Exponents `wᵢ`
    - Linear weights `aᵢ`
    - Scalars `C₁`, `C₂`, etc.

- **Structural Ambiguity**:
  - Different expressions can represent the same function value.
  - This makes optimization harder without proper architectural priors.

---

### 🔧 Design Consideration

- **Positive Intermediate Values**:
  - All intermediate activations are constrained to be positive.
  - Alternative: use `F.softplus()` to allow more flexibility while staying positive.
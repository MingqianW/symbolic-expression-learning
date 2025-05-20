# Symbolic Regression Model Documentation

## Overview

This document records the design and purpose of two neural network models used for symbolic regression:

* `SpecifiedSctureModel1`
* `SpecifiedSctureModel2`

Both models aim to approximate symbolic functions of the form:

$$
f(x) = \underbrace{C_1 \cdot \prod_i \left(\sum_j a_{ij} x_j \right)^{w_i}}_{\text{Term 1}} + \underbrace{\beta^T x}_{\text{Term 2}} + \underbrace{C_2 \cdot \prod_j x_j^{y_j}}_{\text{Term 3}}
$$

## Model 1: `SpecifiedSctureModel1`

### Architecture

* **Term 1**: `PowerActivation(Linear(x))`
* **Term 2**: `Linear(x)` (no bias)
* **Term 3**: `PowerActivation(x)`
* Final output: sum of all three terms

### Code Summary

```python
term1 = self.power_sum(self.linear_sum(x))
term2 = self.linear_term(x)
term3 = self.power_direct(x)
return term1 + term2 + term3
```

### LaTeX Expression

$$
f(x) = \left(\sum_j \beta_j x_j \right) + \gamma_1 \cdot C_1 \cdot \prod_i \left( \sum_j a_{ij} x_j \right)^{w_i} + \gamma_2 \cdot C_2 \cdot \prod_j x_j^{y_j}
$$

---

## Model 2: `SpecifiedSctureModel2`

### Architecture

* **Term 1**: `PowerActivation(Linear(x))`
* **Term 3**: `PowerActivation(x)`
* Final input: `[term1, term3, x]` → passed to a final `Linear` layer (no bias)

### Code Summary

```python
combined = torch.cat([term1, term3, x], dim=1)
return self.final_linear(combined)
```

### LaTeX Expression

$$
f(x) = w_1 \cdot \left( C_1 \cdot \prod_i \left( \sum_j a_{ij} x_j \right)^{w_i} \right)
       + w_2 \cdot \left( C_2 \cdot \prod_j x_j^{y_j} \right)
       + \sum_j \beta_j x_j
$$

---

## Example Target Function

$$
f(x_1, x_2, x_3) = 
1.1 \cdot 
\left(0.2 x_1 + 0.5 x_2 + 0.3 x_3\right)^{0.6} \cdot
\left(0.6 x_1 + 0.1 x_2 + 0.8 x_3\right)^{0.4} 
+ 0.3 x_1 + 0.7 x_2 + 0.2 x_3
+ x_1^{0.5} \cdot x_2^{1.2} \cdot x_3^{0.3}
$$

---

## Final Learned Expressions

### `SpecifiedSctureModel1`

$$
f(x_1, x_2, x_3) = 0.491 \cdot (-0.224 x_1 + 0.351 x_2 + 0.873 x_3)^{0.554} \cdot (0.485 x_1 + 0.042 x_2 + 0.143 x_3)^{0.591} \cdot (0.049 x_1 + 1.077 x_2 - 0.293 x_3)^{0.714} + (0.619 x_1 + 0.913 x_2 + 0.752 x_3) + 0.860 \cdot x_1^{0.512} \cdot x_2^{1.176} \cdot x_3^{0.195}
$$

### `SpecifiedSctureModel2`

$$
f(x_1, x_2, x_3) = 0.841 \cdot (-0.229 x_1 + 0.525 x_2 + 0.978 x_3)^{0.687} \cdot (0.795 x_1 + 0.011 x_2 + 0.088 x_3)^{0.707} \cdot (0.458 x_1 + 1.253 x_2 - 0.091 x_3)^{0.799} + (0.601 x_1 + 1.426 x_2 + 0.598 x_3) - 0.147 \cdot x_1^{1.598} \cdot x_2^{-0.230} \cdot x_3^{1.450}
$$

---

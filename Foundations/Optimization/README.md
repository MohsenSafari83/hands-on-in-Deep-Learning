# 🚀 Optimization in Neural Networks

## Introduction
The foundation of neural network training lies in **Gradient Descent**.  
However, most neural networks are based on **non-convex loss functions** with **highly complex loss surfaces** — containing multiple local minima, saddle points, and flat regions.  

As a result, **vanilla Gradient Descent** often struggles to find the true global minimum:
- It may get trapped in **local minima**.  
- It can **oscillate** in narrow valleys.  
- It depends heavily on a properly chosen **learning rate**.  

To overcome these limitations, advanced optimizers such as **Momentum**, **Adam**, and **Newton’s Method** were developed.

---

## 1️⃣ Gradient Descent — Overview

**Goal:** Minimize the loss function `J(w)` with respect to weights `w`:

`w* = argmin_w J(w)`

**Update Rule:**

`w_{t+1} = w_t - η ∇_w J(w_t)`

**Limitations:**
- Sensitive to the choice of **learning rate** `η`.  
- Can get **stuck** in local minima.  
- **Slow convergence** in flat regions.  
- **Oscillations** in steep or irregular loss surfaces.

---

## 2️⃣ Momentum Optimization

Momentum introduces **memory** into gradient descent — it accumulates the direction of past gradients to build “velocity” that helps escape shallow minima and smooths oscillations.

### First Momentum (Classic Momentum)
**Idea:** Maintain an exponentially decaying moving average of past gradients.

**Update Equations:**

`m_{t+1} = β₁ m_t + (1 - β₁) ∇_w J(w_t)`

`w_{t+1} = w_t - η m_{t+1}`

where `β₁ ∈ [0.9, 0.99]` controls the influence of past gradients.

---

### Second Momentum (Variance Term)
Tracks the **moving average of squared gradients**, controlling step size adaptively:

`v_{t+1} = β₂ v_t + (1 - β₂) (∇_w J(w_t))²`

---

### Bias Correction
To counter early-step bias (since `m₀ = v₀ = 0`):

`m̂_t = m_t / (1 - β₁^t)`  
`v̂_t = v_t / (1 - β₂^t)`

---

### Summary:
- **First Momentum →** direction (velocity)  
- **Second Momentum →** step size (adaptive scaling)  
- Combined, they form the basis for the **Adam optimizer**.

---

## 3️⃣ Adam Optimizer (Adaptive Moment Estimation)

Adam combines both **momentum** and **adaptive learning rate** mechanisms.

**Equations:**

`m_{t+1} = β₁ m_t + (1 - β₁) ∇_w J(w_t)`  
`v_{t+1} = β₂ v_t + (1 - β₂) (∇_w J(w_t))²`

`m̂_{t+1} = m_{t+1} / (1 - β₁^{t+1})`  
`v̂_{t+1} = v_{t+1} / (1 - β₂^{t+1})`

`w_{t+1} = w_t - η * (m̂_{t+1} / (√(v̂_{t+1}) + ε))`

**Typical Parameters:**  
`β₁ = 0.9`, `β₂ = 0.999`, `ε = 10⁻⁸`

**Advantages:**
- Fast convergence  
- Works well with sparse or noisy gradients  
- Minimal hyperparameter tuning required  

---

## 4️⃣ Newton’s Method

Newton’s Method uses **second-order derivatives** to adjust the update direction and step size using curvature information (the Hessian matrix).

**Concept:**
Find `x` where `f'(x) = 0`, then iteratively update:

`x_{t+1} = x_t - f'(x_t) / f''(x_t)`

For optimization in multiple dimensions:

`w_{t+1} = w_t - H⁻¹ ∇_w J(w_t)`

where `H` is the **Hessian matrix** (matrix of second derivatives).

**Pros:**
- Quadratic convergence near minima  
- Adaptive step sizes via curvature  

**Cons:**
- Requires Hessian computation (expensive for large models)  
- Can converge to saddle points in non-convex surfaces  

---

## Summary

| Optimizer | Type | Memory | Adaptivity | Pros | Cons |
|------------|------|---------|-------------|------|------|
| **Gradient Descent** | First-order | Low | ✗ | Simple, intuitive | Sensitive to LR, slow |
| **Momentum** | First-order + history | Medium | ✗ | Smoother, faster convergence | Needs tuning (β₁) |
| **Adam** | First-order adaptive | Medium | ✓ | Fast, robust, adaptive | Sometimes poorer generalization |
| **Newton’s Method** | Second-order | High | ✓ | Fast near minima | Computationally heavy |

---

> ⚡ **In short:**  
> Neural network optimization starts with Gradient Descent, improves with Momentum, adapts with Adam, and theoretically perfects with Newton’s Method — though practical constraints often decide which is used.

# Optimization in Neural Networks

## Introduction
The foundation of neural network training lies in **Gradient Descent**.  
However, neural networks are **non-convex systems** with **complex loss surfaces** full of local minima and saddle points.  
Thus, vanilla Gradient Descent often fails to find global minima, oscillates in narrow valleys, and depends heavily on the learning rate.

---

## 1️. Gradient Descent

**Goal:** minimize loss function \( J(w) \).

**Update Rule:**

<img src="https://latex.codecogs.com/svg.image?w_{t+1}=w_t-\eta\nabla_wJ(w_t)" />

**Limitations:**
- Sensitive to learning rate (η)
- May get stuck in local minima
- Slow in flat regions
- Oscillates in steep valleys

---

## 2️. Momentum Optimization

Momentum introduces **memory** — it accumulates gradients to smooth updates and escape local minima.

### First Momentum (Classic)

**Update Equations:**

<img src="https://latex.codecogs.com/svg.image?m_{t+1}=\beta_1m_t+(1-\beta_1)\nabla_wJ(w_t)" />

<img src="https://latex.codecogs.com/svg.image?w_{t+1}=w_t-\eta m_{t+1}" />

### Second Momentum (Variance Term)

Tracks squared gradients to adjust step sizes adaptively:

<img src="https://latex.codecogs.com/svg.image?v_{t+1}=\beta_2v_t+(1-\beta_2)(\nabla_wJ(w_t))^2" />

### Bias Correction

<img src="https://latex.codecogs.com/svg.image?\hat{m}_t=\frac{m_t}{1-\beta_1^t},\quad\hat{v}_t=\frac{v_t}{1-\beta_2^t}" />

**Summary:**
- First Momentum → direction (velocity)  
- Second Momentum → step size (adaptive)  
- Together form the base of **Adam**

---

## 3️. Adam Optimizer (Adaptive Moment Estimation)

Combines both momenta with adaptive learning rates.

**Equations:**

<img src="https://latex.codecogs.com/svg.image?m_{t+1}=\beta_1m_t+(1-\beta_1)\nabla_wJ(w_t)" />

<img src="https://latex.codecogs.com/svg.image?v_{t+1}=\beta_2v_t+(1-\beta_2)(\nabla_wJ(w_t))^2" />

<img src="https://latex.codecogs.com/svg.image?\hat{m}_{t+1}=\frac{m_{t+1}}{1-\beta_1^{t+1}},\quad\hat{v}_{t+1}=\frac{v_{t+1}}{1-\beta_2^{t+1}}" />

<img src="https://latex.codecogs.com/svg.image?w_{t+1}=w_t-\eta\frac{\hat{m}_{t+1}}{\sqrt{\hat{v}_{t+1}+\epsilon}}" />

**Typical Parameters:**  
β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸

**Advantages:**
- Fast convergence  
- Works with sparse gradients  
- Minimal hyperparameter tuning  

---

## 4️. Newton’s Method

Uses **second-order derivatives (Hessian)** for curvature-based optimization.

**1D Case:**

<img src="https://latex.codecogs.com/svg.image?x_{t+1}=x_t-\frac{f'(x_t)}{f''(x_t)}" />

**Multivariate Form:**

<img src="https://latex.codecogs.com/svg.image?w_{t+1}=w_t-H^{-1}\nabla_wJ(w_t)" />

**Pros:**
- Quadratic convergence  
- Adaptive step sizes via curvature  

**Cons:**
- Expensive Hessian computation  
- High memory usage  
- Can get trapped at saddle points  

---

## Summary

| Optimizer | Type | Memory | Adaptive | Pros | Cons |
|------------|------|---------|-----------|------|------|
| Gradient Descent | First-order | Low | ❌ | Simple | Sensitive to LR |
| Momentum | First-order + history | Medium | ❌ | Smooths updates | Needs tuning |
| Adam | Adaptive + momentum | Medium | ✅ | Fast, stable | Sometimes overfits |
| Newton’s Method | Second-order | High | ✅ | Fast near minima | Expensive |

---

> ⚡ **In short:**  
> Neural network optimization begins with **Gradient Descent**, gains stability with **Momentum**, adapts with **Adam**, and theoretically refines with **Newton’s Method** — though computational cost often limits the latter in deep learning.

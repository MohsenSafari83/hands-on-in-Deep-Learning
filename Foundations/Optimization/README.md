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

**Goal:** Minimize the loss function \( J(w) \) with respect to weights \( w \):

$$
w^{*} = \arg\min_{w} J(w)
$$

**Update Rule:**

$$
w_{t+1} = w_t - \eta \nabla_w J(w_t)
$$

**Limitations:**
- Sensitive to the choice of **learning rate** \( \eta \).  
- Can get **stuck** in local minima.  
- **Slow convergence** in flat regions.  
- **Oscillations** in steep or irregular loss surfaces.

---

## 2️⃣ Momentum Optimization

Momentum introduces **memory** into gradient descent — it accumulates the direction of past gradients to build “velocity” that helps escape shallow minima and smooths oscillations.

### First Momentum (Classic Momentum)
**Idea:** Maintain an exponentially decaying moving average of past gradients.

**Update Equations:**

$$
m_{t+1} = \beta_1 m_t + (1 - \beta_1)\nabla_w J(w_t)
$$

$$
w_{t+1} = w_t - \eta m_{t+1}
$$

where \( \beta_1 \in [0.9, 0.99] \) controls the influence of past gradients.

---

### Second Momentum (Variance Term)
Tracks the **moving average of squared gradients**, controlling step size adaptively:

$$
v_{t+1} = \beta_2 v_t + (1 - \beta_2)(\nabla_w J(w_t))^2
$$

---

### Bias Correction
To counter early-step bias (since \( m_0 = v_0 = 0 \)):

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad 
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

---

### Summary:
- **First Momentum →** direction (velocity)  
- **Second Momentum →** step size (adaptive scaling)  
- Combined, they form the basis for the **Adam optimizer**.

---

## 3️⃣ Adam Optimizer (Adaptive Moment Estimation)

Adam combines both **momentum** and **adaptive learning rate** mechanisms.

**Equations:**

$$
m_{t+1} = \beta_1 m_t + (1 - \beta_1)\nabla_w J(w_t)
$$

$$
v_{t+1} = \beta_2 v_t + (1 - \beta_2)(\nabla_w J(w_t))^2
$$

$$
\hat{m}_{t+1} = \frac{m_{t+1}}{1 - \beta_1^{t+1}}, \quad
\hat{v}_{t+1} = \frac{v_{t+1}}{1 - \beta_2^{t+1}}
$$

$$
w_{t+1} = w_t - \eta \frac{\hat{m}_{t+1}}{\sqrt{\hat{v}_{t+1}} + \epsilon}
$$

**Typical Parameters:**  
\( \beta_1 = 0.9, \ \beta_2 = 0.999, \ \epsilon = 10^{-8} \)

**Advantages:**
- Fast convergence  
- Works well with sparse or noisy gradients  
- Minimal hyperparameter tuning required  

---

## 4️⃣ Newton’s Method

Newton’s Method uses **second-order derivatives** to adjust the update direction and step size using curvature information (the Hessian matrix).

**Concept:**
Find \( x \) where \( f'(x) = 0 \), then iteratively update:

$$
x_{t+1} = x_t - \frac{f'(x_t)}{f''(x_t)}
$$

For optimization in multiple dimensions:

$$
w_{t+1} = w_t - H^{-1}\nabla_w J(w_t)
$$

where \( H \) is the **Hessian matrix** (matrix of second derivatives).

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

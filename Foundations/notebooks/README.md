# Foundations of Deep Learning — Theory Notebooks

This section introduces the **fundamental concepts** behind Deep Learning — starting from the distinction between **Machine Learning (ML)** and **Deep Learning (DL)**, and progressing through **Artificial Neural Networks (ANNs)**, the **Perceptron**, and the **Multi-Layer Perceptron (MLP)**.

It aims to provide both the **intuitive understanding** and **mathematical formulation** of how neural networks process and learn from data.

---

## Topics Covered

### 1. Machine Learning vs Deep Learning

| Machine Learning | Deep Learning |
|:--|:--|
| Learns patterns using statistical models and algorithms | Learns hierarchical representations using artificial neural networks |
| Works effectively with smaller datasets | Requires large amounts of data for effective learning |
| Relies on **manual feature extraction** | Performs **automatic feature extraction** |
| Easier to interpret and debug | More complex, works as a “black box” |
| Can be trained efficiently on CPUs | Often requires GPUs or TPUs |
| Faster training but limited scalability | Slower training but highly scalable and accurate for complex tasks |
 
> Deep Learning is a subset of Machine Learning that uses multi-layered neural networks to automatically learn abstract features from raw data.

---

### 2. What Are Deep Neural Networks (DNNs)?

A **Deep Neural Network (DNN)** extends the idea of a traditional **Artificial Neural Network (ANN)** by adding multiple hidden layers between the input and output.

Each layer transforms the data into higher-level abstractions, allowing the model to learn complex mappings.

**Mathematical Formulation:**

$$
\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}
$$

$$
\mathbf{a}^{(l)} = \sigma(\mathbf{z}^{(l)})
$$

Where:
- \( \mathbf{W}^{(l)} \): Weight matrix of layer \( l \)  
- \( \mathbf{b}^{(l)} \): Bias vector  
- \( \sigma(\cdot) \): Activation function  
- \( \mathbf{a}^{(l-1)} \): Activations from the previous layer  

**Comparison:**

| Concept | ANN | DNN |
|:--|:--|:--|
| Depth | 1–2 hidden layers | Many hidden layers |
| Feature learning | Manual | Automatic |
| Computation | Simpler | More complex |
| Applications | Basic tasks (e.g., regression) | Complex tasks (e.g., image, speech, text) |

---

### 3. The Perceptron — The Building Block

The **Perceptron** is the simplest form of a neural network, consisting of a single neuron that performs a linear combination of inputs and passes the result through an activation function.

**Equation:**

$$
y = f(\mathbf{w}^T \mathbf{x} + b)
$$

Where:
- \( \mathbf{w} \): weight vector  
- \( \mathbf{x} \): input vector  
- \( b \): bias term  
- \( f(\cdot) \): activation (step or sign function)

**Limitations:**
- Can only solve **linearly separable** problems (e.g., fails on XOR)
- No hidden layers → limited representation power

---

### 4. Multi-Layer Perceptron (MLP)

To overcome the Perceptron’s limitations, **Multi-Layer Perceptrons (MLPs)** introduce **hidden layers** and **non-linear activation functions**, enabling the network to learn **non-linear relationships**.

**Forward Propagation:**

$$
\mathbf{a}^{(0)} = \mathbf{x}
$$

$$
\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}
$$

$$
\mathbf{a}^{(l)} = \sigma(\mathbf{z}^{(l)})
$$

**Activation Functions:**
- Sigmoid: \( \sigma(x) = \frac{1}{1 + e^{-x}} \)
- Tanh: \( \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \)
- ReLU: \( \max(0, x) \)

**Loss Function Example:**
- Mean Squared Error (MSE):
  $$
  \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
  $$

**Conceptual Steps:**
1. Initialize weights and biases randomly  
2. Perform forward propagation  
3. Compute the loss  
4. Adjust weights via gradient descent (in later modules)

---

## Sources & References

- [**Deep Learning (DL) vs Machine Learning (ML): A Comparative Guide** — DataCamp](https://www.datacamp.com/tutorial/machine-deep-learning)  
- [**Introduction to Neural Networks** — DataCamp](https://www.datacamp.com/tutorial/introduction-to-deep-neural-networks)  
- [**Multilayer Perceptrons in Machine Learning** — DataCamp](https://www.datacamp.com/tutorial/multilayer-perceptrons-in-machine-learning)  
- [**Neural Networks – A Beginner’s Guide** — GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/neural-networks-a-beginners-guide/)

---

> These sources were used to structure and explain the theoretical concepts presented in the **Foundations** notebooks — including ML vs DL comparison, introduction to DNNs, Perceptron fundamentals, and Multi-Layer Perceptron (MLP) theory.

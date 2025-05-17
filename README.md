# 95% Accuracy All-Numpy MNIST Neural Network

This project implements a fully NumPy-based neural network to classify handwritten digits from the MNIST dataset, achieving a 95% accuracy. It demonstrates fundamental deep learning concepts, including forward and backward propagation, parameter initialization, activation functions, and optimization.

---

## 📝 Project Overview

This repository contains a Python implementation of a multi-layer perceptron (MLP) built from scratch using NumPy. The network is trained on the MNIST dataset to classify images of handwritten chinese digits. The primary goals are:

* Illustrate the mechanics of forward and backward propagation without high-level libraries.
* Understand parameter updates via gradient descent.
* Achieve at least 95% accuracy on the MNIST test set.

---

## 📊 Dataset

The [MNIST dataset](https://www.kaggle.com/datasets/gpreda/chinese-mnist) consists of 15,000 grayscale images (64×64 pixels) of handwritten chinese digits:

* **Training set**: 13,000 images
* **Test set**: 2,000 images
---

## 🏗 Model Architecture

* **Input layer**: 4096 neurons (64×64 pixel values)
* **Hidden layer**: 128 neurons, ReLU activation
* **Output layer**: 15 neurons, Softmax activation

```
Input (4096) → Dense (128, ReLU) → Dense (15, Softmax)
```

---

## ⚙️ Implementation Details

* **Language**: Python 3
* **Core library**: NumPy for matrix operations
* **Key components**:

  * `init_params()`: Initialization for weights, zeros for biases
  * `forward_prop()`: Computes linear combinations and activations for each layer
  * `ReLu()`: Applies the rectified linear unit activation function
  * `softmax()`: Converts logits into probability distributions over classes
  * `back_prop()`: Derives gradients for weights and biases via chain rule
  * `update_params()`: Applies gradient descent updates to parameters

---

## 🚂 Training Process

1. **Load & preprocess data**: Flatten and normalize images.
2. **Initialize parameters** with appropriate scaling.
3. **Loop for \*\*\*\*`n_epochs`**:

   * Forward propagate inputs through layers.
   * Calculate loss & track accuracy.
   * Backpropagate to compute parameter gradients.
   * Update weights/biases with learning rate `α`.

## 📈 Results

After training with mini-batches, Adam optimization, data augmentation, and early stopping, the network achieved:

Best test accuracy: 95.20% (early stop at epoch 228)

Final training accuracy plateau: ~98.7%

Performance snapshot:

Epoch 90  – Train: 96.84% – Test: 92.70%
Epoch 125 – Train: 97.72% – Test: 94.70%
Epoch 175 – Train: 98.52% – Test: 94.00%
Epoch 225 – Train: 98.74% – Test: 93.60%
Early stopping at epoch 228. Best test accuracy: 95.20%


Final test accuracy reaches over 95% with this optimized pipeline.

*Final test accuracy reaches over ****95%**** with this optimized pipeline.*

---

## 🔭 Future Work

- Add additional hidden layers or dropout for regularization
- Experiment with different activation functions and optimizers
- Package as a pip-installable module

---

**Kaleb Coleman**  
Data Science Major, Northern Arizona University

**Inspired by**: Video tutorial by Samson Zhang: https://www.youtube.com/watch?v=w8yWXqWQYmU

Feel free to cite, modify, or contribute via pull requests!

```

# 🐶🐱 Pet Classification using CNN

A deep learning project using TensorFlow and Keras to classify pet images into two categories (e.g., cat vs dog) using a Convolutional Neural Network (CNN) with image augmentation techniques to boost generalization.

---

## 📌 Project Overview

This project implements a CNN-based binary image classifier. The model is trained on an augmented dataset to improve robustness and reduce overfitting. The goal is to strike a balance between high accuracy and computational efficiency.

---

## 📁 Dataset and Preprocessing

### ✅ Data Augmentation (via `ImageDataGenerator`)
To improve training and avoid overfitting, the dataset was augmented with real-time transformations:
- **Rescaling**: Normalizing pixel values from `[0, 255]` to `[0, 1]`
- **Shear Transformation**: Random shear to simulate different perspectives
- **Zoom**: Random zooming into images
- **Horizontal Flip**: Simulating different orientations of pets

### ✅ Data Loading
- Data loaded using `flow_from_directory()`
- Image size: **64x64 pixels**
- Batch size: **32**
- Target class mode: **binary**

---

## 🧱 Model Architecture

| Layer Type         | Details                                                       |
|--------------------|---------------------------------------------------------------|
| **Conv2D**         | 32 filters, (5x5), ReLU, input shape (64, 64, 3)              |
| **MaxPooling2D**   | Pool size (2x2)                                                |
| **Conv2D**         | 64 filters, (5x5), ReLU                                       |
| **MaxPooling2D**   | Pool size (2x2)                                                |
| **Flatten**        | Converts feature maps to a 1D vector                          |
| **Dense**          | 32 units, ReLU                                                |
| **Dropout**        | 40% rate to randomly disable neurons                          |
| **Output Dense**   | 1 unit, **sigmoid** activation for binary classification      |

---

## ⚙️ Model Compilation & Training

- **Loss Function**: `binary_crossentropy`
- **Optimizer**: `Adam`
- **Metrics**: `accuracy`
- **Epochs**: 300
- **Training Feedback**: `verbose=1` to monitor each epoch

---

## 📊 Model Performance

| Epochs | Validation Loss | Validation Accuracy |
|--------|------------------|---------------------|
| 100    | ~1.0773          | ~70%                |
| 200    | ~1.6069          | ~65%                |
| 300    | ~4.1340          | ~60%                |

- The performance shows **overfitting** as training continues past ~100 epochs.
- Validation accuracy **declines** after a certain point even though training continues.

---

## 🧠 Key Learnings

- ✅ **Data Augmentation** is essential for improving generalization with small or limited datasets.
- ✅ **Dropout** reduces overfitting by randomly disabling neurons.
- ⚠️ **Overfitting Warning**: Extended training without early stopping led to reduced accuracy.
- ✅ The ideal number of epochs should be determined by monitoring validation metrics (e.g., early stopping or checkpoints).

---

## 🧰 Technologies Used

| Tool          | Purpose                                 |
|---------------|------------------------------------------|
| Python        | Programming language                     |
| TensorFlow 2  | Deep learning framework                  |
| Keras         | High-level neural network API            |
| NumPy         | Numerical computation                    |
| Matplotlib    | (Optional) Visualization of training data|



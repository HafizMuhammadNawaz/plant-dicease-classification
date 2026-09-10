# Plant Disease Classification using MobileNet 🌿

This repository contains a deep learning project for classifying plant leaf diseases using the **MobileNet** architecture in TensorFlow/Keras. The project includes data preprocessing, image normalization, data augmentation, model training, and comprehensive evaluation (Confusion Matrix, ROC Curve).

## 📊 Dataset Overview
The model is trained on a comprehensive plant disease dataset available on Kaggle:
* **Total Image Folders:** 62 classes (including healthy and diseased plants like Tomato, Potato, Apple, Corn, Rice, etc.)
* **Total Training Images:** ~97,296 images (Split into 80% Training, 20% Validation)
* **Test Images:** 3,461 images

## 🛠️ Project Workflow & Features

### 1. Data Preprocessing & Analytics
* Recursive scanning of directory structures to verify valid extensions (`.png`, `.jpg`, `.jpeg`).
* Calculation of class distributions across 62 categories.

### 2. Image Normalization & Resizing
* Images are dynamically resized to $(224 \times 224)$ pixels to fit MobileNet's default input shape.
* Pixel values are scaled/normalized to the $[0, 1]$ range via `img / 255.0`.

### 3. Advanced Augmentation & Denoising
* **Gaussian Blur:** Applied with a $(5, 5)$ kernel to eliminate sensor noise.
* **Affine Transformations:** Implemented recursive translation shifts (horizontal and vertical shifting by 20% ratio) to improve generalization.

### 4. Model Architecture
* **Base Model:** `MobileNet` initialized without pre-trained weights (`weights=None`) to train from scratch.
* **Custom Top Layers:** Added `Flatten`, a `Dense` layer with 128 units (ReLU activation), a `Dropout` layer (0.5) to avoid overfitting, and a final `Softmax` classification layer.

---

## 🚀 Getting Started

### Prerequisites
Make sure you have the following libraries installed:
```bash
pip install tensorflow numpy matplotlib opencv-python scikit-learn seaborn

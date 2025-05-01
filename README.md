# 🍅 Tomato Leaf Disease Classification Using ResNet50 & MobileNet Feature Extractors with MLP

This repository implements a deep feature extraction and lightweight classification pipeline for detecting tomato leaf diseases using **ResNet50** and **MobileNet** as feature extractors, and a **Multi-Layer Perceptron (MLP)** with three hidden layers as the final classifier.

> 📄 **Based on the SPIE conference paper**:  
> *Comparative Evaluation of ResNet50 and MobileNet Feature Extractors for Tomato Leaf Disease Classification Using MLP*  
> Md. Sami Ul Hoque, et al. — SPIE Defense + Commercial Sensing, 2024

---

## 🎯 Objective

To evaluate and compare ResNet50 and MobileNet as deep feature extractors for plant disease classification using an MLP classifier in terms of:

- Classification performance (accuracy, precision, recall, F1-score)
- Computational efficiency (training time, memory usage)
- Deployment feasibility in edge/mobile environments

---

## 🧪 Methodology

### 📸 Dataset
- 10,000 RGB images of tomato leaves
- 10 classes: 9 disease types + healthy
- Balanced via undersampling and augmentation
- Split: 70% train, 15% validation, 15% test (stratified)

### 🧹 Preprocessing
- Resized to 224×224
- Normalized pixel values [0, 1]
- Augmentations: rotation, flipping, contrast, Gaussian noise

### 🔍 Feature Extraction
- **ResNet50**: 2048D features (after global average pooling)
- **MobileNet**: 1024D features (depthwise separable convolutions)
- Fully connected layers removed

### 📉 Dimensionality Reduction
- **PCA** used to retain 95% variance
- Helps reduce overfitting and computational load

### 🧠 MLP Classifier
- 3 hidden layers: 1024 → 512 → 256 units
- Activations: ReLU, LeakyReLU, Swish
- Dropout: 0.3 | Batch Norm | L2 regularization
- Output layer: Softmax (10 classes)
- Loss: Categorical Crossentropy  
- Optimizer: Adam  
- Epochs: 80 with Early Stopping  
- Learning rate:
  - 0.001 (ResNet50 features)
  - 0.0001 (MobileNet features)

---

## 🧠 Results

| Model      | Accuracy | F1 Score | Training Time | GPU Memory |
|------------|----------|----------|----------------|------------|
| ResNet50   | 92.53%   | 92.54%   | 4.81s (80 epochs) | 2901 MB |
| MobileNet  | 92.00%   | 91.99%   | 0.57s (80 epochs) | 113 MB  |

- 🟢 **ResNet50**: Better accuracy but higher resource use  
- 🟢 **MobileNet**: Slightly lower accuracy, but dramatically more efficient for edge deployment

---

## 📦 Compressed Features

Due to GitHub’s 2GB limit, MobileNet `.npy` features are compressed:

```bash
features/features_mobilenet_compressed.npz
```

Load using:
```python
import numpy as np
data = np.load("features/features_mobilenet_compressed.npz", allow_pickle=True)["data"]
```

---

## 🗂️ Project Structure

```
📦 Tomato_Leaf_Disease
 ┣ 📁 features/
 ┃ ┣ 📄 features_resnet.npy
 ┃ ┣ 📄 features_mobilenet_compressed.npz
 ┃ ┣ 📄 labels.npy
 ┣ 📄 Leaf_Code.ipynb              # Main notebook (training + plots)
 ┣ 📄 compress_npy.py              # Compress .npy to .npz
 ┣ 📄 README.md
 ┗ 📄 requirements.txt             # Python dependencies
```

---

## 🚀 How to Run

### 1. Clone and Install

```bash
git clone https://github.com/RedHood316/Tomato-Leaf-Disease.git
cd Tomato-Leaf-Disease
pip install -r requirements.txt
```

### 2. Launch Notebook

```bash
jupyter notebook Leaf_Code.ipynb
```

---

## 📚 Citation

> Md Sami Ul Hoque, Julian Rene Cuellar Buritica, Al Mahmud, Mahdi Kargar Nigjeh, Robert Leander, and Scott Umbaugh.  
> *Comparative Evaluation of ResNet50 and MobileNet Feature Extractors for Tomato Leaf Disease Classification Using MLP*,  
> SPIE Defense + Commercial Sensing, 2024. [DOI (if applicable)]

---

## 🙏 Acknowledgments

Special thanks to **Dr. Scott Umbaugh** and the **CVIP Lab** at **Southern Illinois University Edwardsville** for their mentorship and support.

---

## 🔮 Future Work

- Fine-tuning MobileNet with attention mechanisms
- Integration with edge devices (Raspberry Pi, Jetson Nano)
- Explainable AI (XAI) visualizations
- Field condition evaluation (occlusion, variable lighting)

---

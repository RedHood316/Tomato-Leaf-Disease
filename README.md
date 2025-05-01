# 🍅 Tomato Leaf Disease Classification Using Deep Feature Extraction and SVM

This project implements a deep learning and machine learning pipeline for classifying **tomato leaf diseases** using pretrained CNNs (ResNet50 and MobileNet) for feature extraction and a **Support Vector Machine (SVM)** for final classification. It is based on the SPIE paper:

> **Comparative Evaluation of ResNet50 and MobileNet Feature Extractors for Tomato Leaf Disease Classification Using MLP**  
> Md. Sami Ul Hoque, et al. — SPIE Defense + Commercial Sensing, 2024

---

## 📄 Overview

This repository demonstrates:
- Deep feature extraction from ResNet50 and MobileNet
- Dataset splitting with stratification
- Classification using a radial basis function (RBF) kernel SVM
- Performance evaluation using accuracy, F1-score, and confusion matrix

The compressed feature files are used to meet GitHub's file size limits.

---

## 🧠 Key Components

- 📂 Dataset: Tomato Leaf Disease dataset (10 classes)
- 🧠 Feature Extractors: ResNet50 and MobileNet (pretrained)
- 🧪 Classifier: SVM with RBF kernel
- 📦 Compressed feature files using `.npz` for GitHub compatibility

---

## 🗂️ Project Structure

```
📦 Tomato_Leaf_Disease
 ┣ 📁 features/
 ┃ ┣ 📄 features_resnet.npy
 ┃ ┣ 📄 features_mobilenet_compressed.npz
 ┃ ┣ 📄 labels.npy
 ┣ 📄 Leaf_Code.ipynb              # Main Jupyter Notebook
 ┣ 📄 compress_npy.py              # Script to compress .npy to .npz
 ┣ 📄 README.md
 ┗ 📄 requirements.txt             # Python dependencies
```

---

## 🚀 How to Run

### 1. Clone the repo

```bash
git clone https://github.com/RedHood316/Tomato-Leaf-Disease.git
cd Tomato-Leaf-Disease
```

### 2. Install required packages

```bash
pip install -r requirements.txt
```

### 3. Run the notebook or script

Use the notebook:

```bash
jupyter notebook Leaf_Code.ipynb
```

Or extract compressed features in Python:

```python
import numpy as np
features = np.load('features/features_mobilenet_compressed.npz', allow_pickle=True)['data']
```

---

## 📊 Evaluation Results

Using **ResNet50 features + SVM**:

- ✅ Accuracy: ~92.5%
- 📈 F1-Score: ~92.5%
- 🧮 Multi-class classification across 10 tomato leaf disease categories

Confusion matrix and classification report are generated using `sklearn` and `seaborn`.

---

## 📦 Large File Notice

⚠️ The original file `features/features_mobilenet.npy` exceeded GitHub's 2GB LFS limit.  
We instead use a compressed version:

```
features/features_mobilenet_compressed.npz (961 MB)
```

You can decompress it using:

```python
data = np.load('features/features_mobilenet_compressed.npz', allow_pickle=True)['data']
```

---

## 🧠 Citation

> Md Sami Ul Hoque, Julian Rene Cuellar Buritica, Al Mahmud, Mahdi Kargar Nigjeh, Robert Leander, and Scott Umbaugh.  
> _Comparative Evaluation of ResNet50 and MobileNet Feature Extractors for Tomato Leaf Disease Classification Using MLP_,  
> SPIE Defense + Commercial Sensing, 2024.

---

## 🙏 Acknowledgments

Special thanks to **Dr. Scott Umbaugh** and the CVIP Lab at **Southern Illinois University Edwardsville** for their guidance and support.

---

## 📌 Future Work

- Add PCA-based feature reduction
- Compare SVM with MLP and decision tree classifiers
- Deploy real-time diagnosis app for edge devices

---

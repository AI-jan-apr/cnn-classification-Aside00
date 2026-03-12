# 🐾 AI Pet Classifier: Cat vs Dog Detection
> **An end-to-end Deep Learning solution achieving 84.26% Validation Accuracy.**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white" />
</p>

---

## 🎯 Project Overview
This project delivers a robust **CNN-based classification pipeline** designed to distinguish between feline and canine features. By leveraging a multi-layer neural network and a modern **Streamlit** interface, it provides a seamless user experience from image upload to high-confidence inference.

---

## 🏗️ Model Architecture
The system follows a hierarchical feature extraction approach to ensure high precision:

| Stage | Component | Configuration | Purpose |
| :--- | :--- | :--- | :--- |
| **Input** | Standard Image | 150x150 RGB | Standardized resolution for processing. |
| **Extraction** | 3x Conv Blocks | 32, 64, 128 Filters | Capturing edges, textures, and patterns. |
| **Non-Linearity**| Activation | `ReLU` | Introducing complexity to feature maps. |
| **Downsampling** | Pooling | `MaxPooling2D` | Reducing compute load & spatial dimensions. |
| **Regularization**| Dropout | `0.5` | **Combatting Overfitting** for generalization. |
| **Output** | Dense Layer | `Sigmoid` | Probability-based classification. |

---

## 📊 Performance Analytics

### 📈 Learning Trajectory
<p align="center">
<img width="483" height="374" alt="image" src="https://github.com/user-attachments/assets/f6c086b8-be77-41ea-94d6-e0418ef671b8" />
</p>

* **Validation Accuracy:** Stabilized at **84.26%**.
* **Insight:** The model shows strong convergence, proving the 3-layer architecture is optimal for this dataset size.

### 📉 Error Optimization
<p align="center">
<img width="277" height="435" alt="image" src="https://github.com/user-attachments/assets/2917d947-d476-4f0e-b7d3-caedc17f1f2a" />
</p>

* **Training Loss:** Successfully minimized to **0.14**.
* **Optimal Point:** Reach peak efficiency at **Epoch 5** (Val Loss: 0.3715).

---

## 🧪 Production-Ready Validation
Real-world testing with unseen pet images yielded the following results:

> [!IMPORTANT]
> - **🐕 Dog Detection:** Identified with **91.0% Confidence**
> - **🐈 Cat Detection:** Identified with **93.0% Confidence**

---

<img width="1865" height="910" alt="image" src="https://github.com/user-attachments/assets/68bfd39c-1af2-4b59-bf5b-c5805c159af0" />


## 🛠️ Tech Stack & Environment
* **Core Engine:** `TensorFlow` & `Keras`
* **Image Processing:** `Pillow` (PIL) & `NumPy`
* **UI/UX:** `Streamlit` with Custom Modern CSS
* **Development:** `Google Colab` & `VS Code`

---

## 🚀 Quick Start
1. **Clone the repository:**
   ```bash
   git clone [https://gitlab.com/your-username/your-repo.git](https://gitlab.com/your-username/your-repo.git)

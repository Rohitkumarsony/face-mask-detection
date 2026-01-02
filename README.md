# 😷 Face Mask Detection using CNN

This project implements a **computer vision–based face mask detection system** using a **Convolutional Neural Network (CNN)** with **transfer learning**. The model classifies whether a person is wearing a face mask or not from an input image.

---

## 🎯 Objective
To classify face images into two categories:
- **With Mask**
- **Without Mask**

Such systems are commonly used in **public safety**, **workplace compliance**, and **automated monitoring/surveillance solutions**.

---

## 🚀 Features
- Binary image classification (**Mask / No Mask**)
- CNN built using **MobileNetV2 (Transfer Learning)**
- Automated dataset download using **KaggleHub**
- Supports **URL-based image prediction**
- Lightweight and efficient inference
- Fully reproducible using **Google Colab**

---

## 🧠 Tech Stack
- Python  
- TensorFlow / Keras  
- NumPy  
- Pillow  
- Matplotlib  
- KaggleHub  



---

## 📊 Dataset
**Face Mask Dataset (Kaggle)**  
- Two classes: `with_mask`, `without_mask`
- Dataset is downloaded programmatically using **KaggleHub**

**Dataset Source:**  
https://www.kaggle.com/datasets/omkargurav/face-mask-dataset

---

## 🧪 Model Architecture
- Pretrained **MobileNetV2** used as the feature extractor
- Global Average Pooling layer
- Fully Connected (Dense) output layer
- **Sigmoid activation** for binary classification

---

## 🧠 Training Configuration
- Image size: `224 × 224`
- Optimizer: **Adam**
- Loss function: **Binary Crossentropy**
- Epochs: `5–8`
- Data augmentation applied during training to improve generalization

---

## 🔍 Inference
The trained model supports **URL-based image prediction**:
1. Provide an image URL
2. Image is downloaded and preprocessed
3. CNN predicts **With Mask / Without Mask**
4. Prediction confidence score is displayed

---

## ▶️ How to Run (Google Colab)
1. Open the Google Colab notebook
2. Install required dependencies
3. Download the dataset using **KaggleHub**
4. Train the CNN model
5. Run inference using image URLs

📌 **Google Colab Notebook:**  
https://colab.research.google.com/drive/1ttUipqEJpxNwbxwsAqoIPoFQB1xLoW6m#scrollTo=Vf5YK_5aRygG

---

## 📝 Notes
- Streamlit UI has been removed from this project.
- The focus is on **model training and inference**, not UI development.
- The trained `.h5` model can be reused for deployment or integration into other applications.
- This project is designed as a clean and reproducible **Proof of Concept (PoC)** for face mask detection.

---

## 👨‍💻 Author
**Rohit Kumar**

---

# Avengers Face Recognition using Deep Learning

This project focuses on building a deep learning–based face recognition system to identify Avengers characters from images. The system learns distinctive facial features of each character and classifies input images into the correct Avengers category. The project explores different learning behaviors and evaluates model performance based on accuracy and generalization to unseen images.

---

## Problem Statement

The objective of this project is to develop a system capable of recognizing and classifying images of Avengers characters. The task involves learning unique facial features from labeled image data while handling variations in pose, lighting, expressions, and image quality. Model performance is evaluated based on classification accuracy and robustness on unseen images.

---

## 🗂 Dataset Description

* Dataset consists of facial images of multiple Avengers characters
* Images are organized in a directory-based class structure
* 5 claases are there and each class represents one Avengers character

---

## Project Workflow

1. Data loading and preprocessing using image generators
2. Training multiple deep learning models to study different learning behaviors
3. Performance evaluation using accuracy, loss, confusion matrix, and classification report
4. Testing generalization using random unseen images
5. Comparison of model performance to identify the most effective approach

---

## Models Implemented

The project experiments with three deep learning approaches:

* A high-capacity CNN to analyze overfitting behavior
* A shallow CNN to demonstrate underfitting behavior
* A transfer learning–based model leveraging pretrained feature extraction

These models help in understanding the impact of model complexity and generalization in face recognition tasks.

---

## Evaluation Metrics

Model performance is evaluated using:

* Training and validation accuracy
* Training and validation loss
* Confusion matrix
* Precision, recall, and F1-score (classification report)
* Visual comparison of predicted vs actual labels on random test images

---

## Key Observations

* High-capacity models tend to overfit when regularization is insufficient
* Shallow models fail to capture complex facial features and underperform
* Transfer learning improves generalization and stability
* Pretrained feature extractors significantly reduce training time and improve accuracy
* Best model shows consistent performance on unseen Avengers images

---

## Final Conclusion

The project demonstrates that model complexity and feature extraction play a critical role in face recognition tasks. While simpler models struggle to learn meaningful patterns, overly complex models risk overfitting. Transfer learning provides the most balanced performance, achieving higher accuracy and better generalization on unseen Avengers character images.

---

## Sample Predictions

The project includes functionality to:

* Randomly select images from the test set
* Display predicted labels alongside actual labels
* Visually analyze correct and incorrect predictions

This helps in understanding real-world performance beyond numerical metrics.

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## How to Run the Project

1. Clone the repository

   ```bash
   git clone https://github.com/Ruchii151/avengers-character-face-recognition.git
   ```
2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook

   ```bash
   jupyter notebook
   ```
4. Run all cells sequentially

---

## Future Improvements

* Increase dataset size for better generalization
* Apply data augmentation techniques more aggressively
* Experiment with face alignment and detection before classification
* Deploy the model as a web application

---


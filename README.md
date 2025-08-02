# Handwritten Digit Recognition System

This project implements a comprehensive **Handwritten Digit Recognition** system using the **MNIST** dataset. It evaluates multiple machine learning and deep learning models including CNNs, SVMs, k-NN, and Random Forests, while employing feature extraction techniques like **HOG** and **PCA**.

---

## Features

- Uses the MNIST dataset of handwritten digits (28x28 grayscale images)
- Preprocessing: normalization, Gaussian blur, and data augmentation
- Feature extraction:
  - Raw pixel intensities
  - Histogram of Oriented Gradients (HOG)
  - Principal Component Analysis (PCA)
- Model Training & Evaluation:
  - Convolutional Neural Network (CNN)
  - Support Vector Machine (SVM)
  - k-Nearest Neighbors (k-NN)
  - Random Forest Classifier
- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion matrix and classification report
- Visualization:
  - Confusion matrices
  - Model predictions on test samples
- Real-time digit prediction using saved CNN model and custom images

---

## Installation


pip install -r requirements.txt
Usage
Training & Evaluation
Run the Python script to preprocess data, extract features, train models, and evaluate their performance.

Prediction on Custom Image


predicted_digit = predict_image("path/to/image.png")
visualize_prediction("path/to/image.png")
Model Saving
The CNN model is saved to disk after training:


/content/drive/MyDrive/handwritten Digit Recognition Model/model.h5
File Structure
Copy
Edit
├── Handwritten Digit Recognition.py
├── requirements.txt
└── README.md
Dataset
The MNIST dataset is used, consisting of:

60,000 training images

10,000 testing images

Technologies Used
Python

TensorFlow / Keras

scikit-learn

OpenCV

scikit-image

matplotlib

License
This project is for educational and research purposes.


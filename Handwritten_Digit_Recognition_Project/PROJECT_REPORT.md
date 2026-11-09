# Handwritten Digit Recognition - Project Report

## 1. Introduction

Handwritten Digit Recognition is a machine learning application that identifies numerical digits written by a person. The project uses the MNIST dataset and a Convolutional Neural Network (CNN).

## 2. Objective

The main objective is to build a deep learning model that can recognize handwritten digits from 0 to 9 and provide the predicted digit through a simple web interface.

## 3. Dataset

The project uses the MNIST dataset. It contains grayscale images of handwritten digits. Each image has a size of 28 × 28 pixels and belongs to one of ten classes: 0, 1, 2, 3, 4, 5, 6, 7, 8, and 9.

## 4. Methodology

1. Load the MNIST dataset.
2. Normalize pixel values between 0 and 1.
3. Reshape images for CNN input.
4. Build a CNN with convolution and pooling layers.
5. Train the model.
6. Evaluate the model on test data.
7. Save the trained model.
8. Create a Streamlit interface.
9. Accept a drawn or uploaded digit.
10. Preprocess the input and predict the digit.

## 5. CNN Model

The model contains:

- Input layer
- Convolutional layer
- Max pooling layer
- Second convolutional layer
- Second max pooling layer
- Flatten layer
- Dense layer
- Dropout layer
- Output layer with 10 classes

The final layer uses softmax activation to produce probabilities for digits 0-9.

## 6. Evaluation

The model is evaluated using test accuracy. The exact accuracy is written to `training_result.txt` after training because it depends on the training run and environment.

## 7. User Interface

The Streamlit application allows the user to draw a digit or upload an image. After clicking the prediction button, the application displays the predicted digit and confidence.

## 8. Applications

- Automatic form processing
- Digit recognition
- Postal code recognition
- Educational applications
- OCR-related systems
- Document digitization

## 9. Limitations

The model is trained on MNIST-style images. Images that are very different from MNIST, unclear, rotated, or containing multiple digits may reduce prediction accuracy.

## 10. Future Scope

The system can be extended to:

- Recognize handwritten alphabets
- Recognize multiple digits
- Recognize complete words
- Use CRNN or other sequence models
- Deploy the application online

# Handwritten Digit Recognition using CNN

This project recognizes handwritten digits (0-9) using the MNIST dataset and a Convolutional Neural Network (CNN).

## Project files

- `train_model.py` - downloads MNIST, trains the CNN and saves `mnist_cnn.keras`
- `app.py` - Streamlit web application
- `requirements.txt` - required Python libraries
- `training_result.txt` - created after model training

## How to run in VS Code

### Step 1: Open the project folder

Open this folder in VS Code:

`Handwritten_Digit_Recognition_Project`

### Step 2: Open Terminal

In VS Code select:

Terminal → New Terminal

### Step 3: Install libraries

Run:

```bash
pip install -r requirements.txt
```

If `pip` is not recognized, try:

```bash
python -m pip install -r requirements.txt
```

### Step 4: Train the model

Run:

```bash
python train_model.py
```

The first run downloads the MNIST dataset and trains the CNN. When training finishes, you should see:

`Model saved as: mnist_cnn.keras`

### Step 5: Start the application

Run:

```bash
streamlit run app.py
```

A browser page should open automatically.

### Step 6: Test

Draw a digit such as 5 or 8 on the canvas and click:

`Predict Drawn Digit`

You can also upload a digit image.

## Important

The model file `mnist_cnn.keras` is generated after running `train_model.py`. It is not included in the starter project because it is a generated model file.

## Technologies

- Python
- TensorFlow
- CNN
- MNIST
- Streamlit
- NumPy
- Pillow

## Objective

To identify handwritten digits using image processing and deep learning.

## Expected output

The application displays:

- Predicted digit
- Confidence percentage
- Warning when the model is uncertain

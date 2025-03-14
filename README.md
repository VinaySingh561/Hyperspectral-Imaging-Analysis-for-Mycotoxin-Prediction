# Hyperspectral Imaging Analysis for Mycotoxin Prediction

## Project Overview

This repository contains my submission for the ML Intern assignment that focuses on processing hyperspectral imaging data to predict mycotoxin (DON/vomitoxin) levels in corn samples. The project implements both a Deep Neural Network (DNN) and a Convolutional Neural Network (CNN) approach to address the regression task of predicting DON concentration from spectral reflectance data.

Hyperspectral imaging provides rich spectral information across multiple wavelength bands, and this project demonstrates how to leverage this data through dimensionality reduction techniques and deep learning models to achieve accurate predictions of mycotoxin levels, which is crucial for food safety and quality control in agriculture.

## Repository Structure

```
├── DNNmodel.py            # Deep Neural Network implementation 
├── CNN_model.ipynb        # Convolutional Neural Network implementation in Jupyter Notebook
├── Task-for-ML-Intern.docx # Original assignment description
├── README.md              # This file
```

## Models Implemented

### 1. Deep Neural Network (DNNmodel.py)

The DNN model uses a multi-layer perceptron architecture with dropout regularization to predict DON concentration from PCA-reduced hyperspectral data. The implementation includes:

- Data loading and preprocessing with standardization
- Principal Component Analysis (PCA) for dimensionality reduction
- A neural network with multiple dense layers and dropout for regularization
- Model training and evaluation with comprehensive metrics
- Visualization of prediction results

The DNN architecture consists of three hidden layers (64, 32, and 16 neurons) with ReLU activation functions and dropout layers to prevent overfitting.

### 2. Convolutional Neural Network (CNN_model.ipynb)

The CNN approach treats the spectral bands as a one-dimensional sequence, applying 1D convolutions to capture patterns across the wavelength domain. Key components include:

- Data reshaping for CNN input (samples, channels, sequence length)
- 1D convolutional layers with batch normalization
- Max pooling for feature reduction
- Fully connected layers for final regression
- Extensive training with Adam optimizer
- Performance evaluation with MAE, RMSE, and R² metrics

The CNN model achieved an R² score of 0.7768, indicating strong predictive performance.

## Installation Instructions

1. Clone this repository:
```bash
git clone https://github.com/yourusername/hyperspectral-mycotoxin-prediction.git
cd hyperspectral-mycotoxin-prediction
```

2. Create a virtual environment (recommended):
```bash
python -m venv env
source env/bin/activate  # On Windows, use: env\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install numpy pandas matplotlib scikit-learn tensorflow torch
```

## Usage Instructions

### Running the DNN Model:

```bash
python DNNmodel.py path/to/dataset.csv
```

The script will:
1. Load and preprocess the data
2. Apply PCA for dimensionality reduction
3. Train the DNN model
4. Evaluate model performance
5. Visualize actual vs. predicted values

### Running the CNN Model:

Open the Jupyter notebook:
```bash
jupyter notebook CNN_model.ipynb
```

Execute each cell sequentially to:
1. Load and preprocess the dataset
2. Define the CNN architecture 
3. Train the model for the specified epochs
4. Visualize training loss
5. Evaluate model performance with metrics and plots

## Results and Evaluation

### DNN Model
The DNN model leverages PCA to reduce dimensionality while preserving variance in the spectral data. Evaluation metrics are calculated to assess model performance, including Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² score.

### CNN Model
The CNN model achieved:
- MAE: 2618.5415
- RMSE: 7899.5033
- R²: 0.7768

The R² score of approximately 0.78 indicates that the model explains about 78% of the variance in DON concentration, demonstrating good predictive capability. The scatter plot of actual vs. predicted values shows a strong correlation with some variance at higher concentration levels.

## Future Improvements

Potential enhancements for the models include:
- Implementing attention mechanisms or transformers for improved performance
- Exploring ensemble methods combining multiple models
- Further hyperparameter optimization
- Feature importance analysis to identify key wavelength bands
- Developing a Streamlit application for interactive predictions from user-uploaded spectral data

## Assignment Requirements

This project satisfies the requirements specified in the ML Intern task:
- Data preprocessing and normalization
- Dimensionality reduction using PCA
- Model training using both DNN and CNN approaches
- Comprehensive evaluation metrics (MAE, RMSE, R²)
- Visualization of results with scatter plots
- Clean, modular, and well-commented code

## Contact

For any questions or feedback regarding this project, please contact:
[Vinay Singh] - [vs1739561@gmail.com]

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/51139083/7d388284-5981-4dbb-94c7-98db17b95d98/DNNmodel.py
[2] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/51139083/4ed8ce3b-1344-49a8-9cfa-959453b2c6ea/Task-for-ML-Intern.docx
[3] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/51139083/f282d968-a1dd-48a9-b2f0-ac04e18400e9/CNN_model.ipynb


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Step 1: Load and preprocess the data
def load_and_preprocess_data(filepath):
    # Load dataset
    data = pd.read_csv(filepath)
    X = data.iloc[:, 1:-1]  # Exclude ID and target
    y = data['vomitoxin_ppb']

    # Handle missing values (if any)
    if X.isnull().any().any():
        X = X.fillna(X.mean())  # Fill missing values with column mean

    # Normalize/standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

# Step 2: Dimensionality reduction using PCA
def apply_pca(X, n_components=10):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    return X_pca

# Step 3: Train a Deep Neural Network (DNN)
def build_and_train_dnn(X_train, y_train, X_test, y_test):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(16, activation='relu'),
        Dense(1)  # Output layer for regression
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=5000, batch_size=32, validation_split=0.2, verbose=1)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

    return model, history, y_pred

# Step 4: Visualize results
def visualize_results(y_test, y_pred):
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual DON Concentration")
    plt.ylabel("Predicted DON Concentration")
    plt.title("Actual vs. Predicted DON Concentration")
    plt.show()

# Main function
def main(filepath):
    # Step 1: Load and preprocess data
    X, y = load_and_preprocess_data(filepath)

    # Step 2: Apply PCA for dimensionality reduction
    X_pca = apply_pca(X, n_components=10)

    # Step 3: Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    # Step 4: Train the DNN model
    model, history, y_pred = build_and_train_dnn(X_train, y_train, X_test, y_test)

    # Step 5: Visualize results
    visualize_results(y_test, y_pred)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python task_ml_intern.py <path_to_dataset.csv>")
    else:
        main(sys.argv[1])
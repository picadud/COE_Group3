import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import stage0_configs 
import stage3_prepare_simCurves


def extract_features_and_labels(FD_curves):
    """
    This function takes the FD curves and extracts features (displacement, force)
    and labels (peaking and dropping points) for model training.
    """
    X, y_peak, y_drop = [], [], []
    
    for paramsTuple, dispForce in FD_curves.items():
        displacement = dispForce["displacement"]
        force = dispForce["force"]

        # Extract features: here we use displacement as the feature for simplicity
        features = np.array(displacement).reshape(-1, 1)

        # Find the peaking point (maximum force)
        peak_index = np.argmax(force)
        peak_displacement = displacement[peak_index]
        peak_force = force[peak_index]

        # Find the dropping point (after the peak where force first decreases significantly)
        drop_index = peak_index + np.argmin(force[peak_index:]) if len(force[peak_index:]) > 0 else len(force)-1
        drop_displacement = displacement[drop_index]
        drop_force = force[drop_index]

        # Append the data
        X.append(features)
        y_peak.append([peak_displacement, peak_force])
        y_drop.append([drop_displacement, drop_force])

    return np.vstack(X), np.array(y_peak), np.array(y_drop)

def train_predict_peak_drop(info, FD_curves_dict):
    """
    This function trains a machine learning model to predict the peaking and dropping points
    in FD curves and applies the model to make predictions on new curves.
    """
    combined_FD_curves = FD_curves_dict['combined_objective_value_to_param_FD_Curves']

    # Extract features (displacement) and labels (peaking and dropping points)
    X, y_peak, y_drop = extract_features_and_labels(combined_FD_curves)

    # Split data into training and test sets
    X_train, X_test, y_train_peak, y_test_peak = train_test_split(X, y_peak, test_size=0.2, random_state=42)
    _, _, y_train_drop, y_test_drop = train_test_split(X, y_drop, test_size=0.2, random_state=42)

    # Initialize and train Random Forest model for peaking point prediction
    model_peak = RandomForestRegressor(n_estimators=100, random_state=42)
    model_peak.fit(X_train, y_train_peak)

    # Initialize and train Random Forest model for dropping point prediction
    model_drop = RandomForestRegressor(n_estimators=100, random_state=42)
    model_drop.fit(X_train, y_train_drop)

    # Predict on test data
    y_pred_peak = model_peak.predict(X_test)
    y_pred_drop = model_drop.predict(X_test)

    # Calculate errors (MSE)
    mse_peak = mean_squared_error(y_test_peak, y_pred_peak)
    mse_drop = mean_squared_error(y_test_drop, y_pred_drop)

    print(f"Mean Squared Error for Peaking Point Prediction: {mse_peak}")
    print(f"Mean Squared Error for Dropping Point Prediction: {mse_drop}")

    # Visualization (optional)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(y_test_peak[:, 0], y_test_peak[:, 1], color='blue', label="True Peak Points")
    plt.scatter(y_pred_peak[:, 0], y_pred_peak[:, 1], color='red', label="Predicted Peak Points")
    plt.title("True vs Predicted Peaking Points")
    plt.xlabel("Displacement (nm)")
    plt.ylabel("Force (μN)")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(y_test_drop[:, 0], y_test_drop[:, 1], color='blue', label="True Drop Points")
    plt.scatter(y_pred_drop[:, 0], y_pred_drop[:, 1], color='red', label="Predicted Drop Points")
    plt.title("True vs Predicted Dropping Points")
    plt.xlabel("Displacement (nm)")
    plt.ylabel("Force (μN)")
    plt.legend()

    plt.tight_layout()
    plt.show()

    return model_peak, model_drop

if __name__ == "__main__":
    # Retrieve combined FD curves from Step 3
    info = stage0_configs.main_config()
    FD_curves_dict = stage3_prepare_simCurves(info)

    # Train and predict peaking and dropping points
    train_predict_peak_drop(info, FD_curves_dict)

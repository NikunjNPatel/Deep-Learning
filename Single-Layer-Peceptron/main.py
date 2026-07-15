import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SingleLayerPerceptrons import SingleLayerPerceptrons

def main():
    # Generate synthetic data
    w_true = 2.5
    b_true = 1.0
    X, Y = generate_synthetic_data(w_true, b_true, n_samples=200, noise=2)

    # Split into training and testing sets
    split_ratio = 0.8
    split_index = int(split_ratio * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    Y_train, Y_test = Y[:split_index], Y[split_index:]

    # Initialize and train the model
    model = SingleLayerPerceptrons(learning_rate=0.01, n_iterations=500)
    model.fit(X_train, Y_train)

    # Display model weights and bias
    print(f"Learned weights: {model.weights.flatten()}")
    print(f"Learned bias: {model.bias}")

    # Predict on data
    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)

    # Generate metrics
    metrics_train = model.compute_metrics(Y_train, Y_train_pred)
    metrics_test = model.compute_metrics(Y_test, Y_test_pred)

    # Display metrics
    print("Training Metrics:")
    for metric, value in metrics_train.items():
        print(f"  {metric}: {value:.4f}")
    print("Testing Metrics:")
    for metric, value in metrics_test.items():
        print(f"  {metric}: {value:.4f}")

    # Plot results
    plot_results(model, X_train, Y_train, X_test, Y_test)
    plot_loss_curve(model)
    plt.show()

def plot_results(model, X_train, Y_train, X_test, Y_test):
    """
    Plot the training and testing results of the model.
    """
    
    fig = plt.figure(figsize=(12, 8))

    # Prediction vs Actual Training Data
    ax2 = fig.add_subplot(2,2,1)
    y_train_pred = model.predict(X_train)
    ax2.scatter(X_train, Y_train, color='blue', label='Actual', alpha=0.5)

    min_value = min(Y_train.min(), y_train_pred.min())
    max_value = max(Y_train.max(), y_train_pred.max())
    ax2.plot(X_train, y_train_pred, 'r--', label='Perfect Prediction')

    ax2.set_xlabel("Actual data")
    ax2.set_ylabel("Predicted data")
    ax2.set_title("Training Data: Actual vs Predicted", pad=10)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Prediction vs Actual Testing Data
    ax3 = fig.add_subplot(2,2,2)
    y_test_pred = model.predict(X_test)
    ax3.scatter(X_test, Y_test, color='green', label='Actual', alpha=0.5)

    min_value = min(Y_test.min(), y_test_pred.min())
    max_value = max(Y_test.max(), y_test_pred.max())
    ax3.plot(X_test, y_test_pred, 'r--', label='Perfect Prediction')
    ax3.set_xlabel("Actual data")
    ax3.set_ylabel("Predicted data")
    ax3.set_title("Testing Data: Actual vs Predicted", pad=10)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Residuals for Training Data
    ax4 = fig.add_subplot(2,2,3)
    residuals_train = Y_train - y_train_pred
    ax4.scatter(y_train_pred, residuals_train, color='blue', alpha=0.5)
    ax4.axhline(y=0, color='r', linestyle='--')
    ax4.set_xlabel("Predicted data")
    ax4.set_ylabel("Residuals")
    ax4.set_title("Training Data: Residuals", pad=10)
    ax4.grid(True, alpha=0.3)

    # Residuals for Testing Data
    ax5 = fig.add_subplot(2,2,4)
    residuals_test = Y_test - y_test_pred
    ax5.scatter(y_test_pred, residuals_test, color='green', alpha=0.5)
    ax5.axhline(y=0, color='r', linestyle='--')
    ax5.set_xlabel("Predicted data")
    ax5.set_ylabel("Residuals")
    ax5.set_title("Testing Data: Residuals", pad=10)
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()

def plot_loss_curve(model):
    # Loss Curve
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(model.loss_history, color='blue', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Curve', pad=10)
    ax1.grid(True, alpha=0.3)

def generate_synthetic_data(w_true, b_true,n_samples=100, noise=0.1):
    """
    Generate synthetic linear data with noise.
    """
    np.random.seed(42)
    X = np.random.rand(n_samples, 1) * 10  # Features between 0 and 10
    Y = w_true * X + b_true + np.random.randn(n_samples, 1) * noise  # Linear relation with noise
    return X, Y

if __name__ == "__main__":
    main()
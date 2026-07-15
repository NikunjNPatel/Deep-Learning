import numpy as np

class SingleNeuronRegressor:
    """
    A simple single neuron regression model with identity activation function.
    """
    def __init__(self,learning_rate = 0.01, n_iterations = 1000, random_seed = 42):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_seed = random_seed
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def add_bias(self, X):
        """
        Add a bias term (column of ones) to the input features.
        Concatenates a column of ones to the input feature matrix X.
        X = [1, x1, x2, ..., xn]
        """
        return np.c_[np.ones(X.shape[0]), X]
    
    def initialize_parameters(self, n_features):
        """
        Initialize weights and bias to small random values.
        """
        np.random.seed(self.random_seed)
        self.weights = np.zeros((n_features, 1))
        self.bias = 0.0

    def predict(self, X):
        """
        Predict the output for given input features X.
        y^ = X * w + b
        """
        return np.dot(X, self.weights) + self.bias

    def compute_loss(self, Y_true, Y_pred):
        """
        Compute Mean Squared Error loss.
        J(w,b) = (1/2*N) * sum((y^ - y)^2)
        """
        N = Y_true.shape[0]
        return (1/(2*N))*np.sum((Y_pred - Y_true) ** 2)
    
    def compute_gradients(self, X, Y_true, Y_pred):
        """
        Compute gradients of the loss with respect to weights and bias.
        dJ/dw = (1/N) * X^T * (y^ - y)
        dJ/db = (1/N) * sum(y^ - y)
        """
        N = Y_true.shape[0]
        error = Y_pred - Y_true
        dw = (1/N) * np.dot(X.T, error)
        db = (1/N) * np.sum(error)
        return dw, db
    
    def fit(self, X, Y, verbose=True):
        """
        Train the model using gradient descent.
        """
        N, n_features = X.shape
        self.initialize_parameters(n_features)

        for i in range(self.n_iterations):
            # Forward pass: Predict the output
            Y_pred = self.predict(X)

            # Compute the loss and store it
            loss = self.compute_loss(Y, Y_pred)
            self.loss_history.append(loss)

            # Backward pass: Compute gradients and update weights and bias
            dw, db = self.compute_gradients(X, Y, Y_pred)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if verbose and (i % 100 == 0  and i == self.n_iterations - 1):
                print(f"Iteration {i:4d}, Loss: {loss:.4f}")
        
        if verbose:
            print("Training complete.")
            print(f"Final Loss: {self.loss_history[-1]:.4f}")
    
    def compute_metrics(self, Y_true, Y_pred):
        """
        Compute evaluation metrics: 
        MSE = (1/N) * sum((y^ - y)^2) # Mean Squared Error
        RMSE = sqrt(MSE) # Root Mean Squared Error
        MAE = (1/N) * sum(|y^ - y|) # Mean Absolute Error
        R2 = 1 - (sum((y^ - y)^2) / sum((y - mean(y))^2)) # R2 Score
        """
        mse = np.mean((Y_pred - Y_true) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(Y_pred - Y_true))
        r2 = 1 - (np.sum((Y_pred - Y_true) ** 2) / np.sum((Y_true - np.mean(Y_true)) ** 2))
        return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}
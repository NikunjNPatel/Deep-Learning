## Linear Regression with a Single Layer Perceptron Model

Linear regression is a fundamental supervised learning algorithm for predicting continuous values. In this implementation, the model is built as a single layer with an identity activation function, learning a linear relationship between input features and the target variable.

### 1. Model Architecture

- **Input Layer:** Feature vector with $d$ features
- **Single Output Neuron:** Linear activation (identity function)
- **No Hidden Layers**

### 2. Hypothesis Function

The model predicts target values using a linear function:

$$
\hat{y} = Xw + b
$$

Where:

- $X$ is the input matrix of shape $(N, d)$
- $w$ is the weight vector of shape $(d, 1)$
- $b$ is the scalar bias term
- $\hat{y}$ is the predicted output vector of shape $(N, 1)$

### 3. Loss Function

We use Mean Squared Error (MSE) as the loss for regression. The training objective is to minimize:

$$
J(w, b) = \frac{1}{2N} \sum_{i=1}^{N} (\hat{y}^{(i)} - y^{(i)})^2
$$

Where $N$ is the number of training examples.

### 4. Gradient Descent

The weights and bias are updated with gradient descent using the following derivatives:

$$
\frac{\partial J}{\partial w} = \frac{1}{N} X^T (\hat{y} - y)
$$

$$
\frac{\partial J}{\partial b} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}^{(i)} - y^{(i)})
$$

Parameter updates:

$$
w \leftarrow w - \alpha \frac{\partial J}{\partial w}
$$

$$
b \leftarrow b - \alpha \frac{\partial J}{\partial b}
$$

Where $\alpha$ is the learning rate.

### 5. Implementation Details

The `SingleNeuronRegressor` class in `SingleNeuronRegressor.py` includes:

- `__init__`: initialize learning rate, iterations, and random seed
- `initialize_parameters`: set initial weights and bias
- `predict`: compute predictions using $Xw + b$
- `compute_loss`: calculate the MSE loss
- `compute_gradients`: compute weight and bias gradients
- `fit`: train the model using batch gradient descent
- `compute_metrics`: evaluate predictions with MSE, RMSE, MAE, and $R^2$

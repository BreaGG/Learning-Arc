# How Machines Learn

> *Breaking down the learning process into fundamental concepts and mathematical representations.*

### 1. Understanding Data and Features

In machine learning, the starting point is always the **data**. Data is typically represented as a matrix, where each row corresponds to an example (data point) and each column represents a feature (attribute) of that example:

$$
  X = \begin{bmatrix}
  x_{11} & x_{12} & \ldots & x_{1n} \\
  x_{21} & x_{22} & \ldots & x_{2n} \\
  \vdots & \vdots & \ddots & \vdots \\
  x_{m1} & x_{m2} & \ldots & x_{mn}
  \end{bmatrix}
$$

Where:
- \(X\) is the **feature matrix**,
- \(m\) is the number of examples (rows),
- \(n\) is the number of features (columns).

Each row represents a single data point with multiple features.

### 2. Learning Objective

The goal of machine learning is to find a function \(f\) that maps input features \(X\) to an output \(y\). This is called **function approximation**.

In **supervised learning**, we aim to predict a target value \(y\) based on the input data \(X\):

$$
y = f(X) + \epsilon
$$

Where:
- \(y\) is the **target value** (the ground truth),
- \(f(X)\) is the **predicted value**,
- \(\epsilon\) is the **error term** (representing noise or uncertainty).

### 3. Model Representation

One of the simplest models is a **linear model**, where the predicted value is a linear combination of the input features. Mathematically, this can be represented as:

$$
\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_n x_n
$$

Where:
- \(\hat{y}\) is the **predicted value**,
- \(\beta_0, \beta_1, \ldots, \beta_n\) are the model parameters (coefficients),
- \(x_1, x_2, \dots, x_n\) are the input features.

### 4. Loss Function

To **measure error**, we use a loss function that quantifies the difference between the predicted values and the actual target values:

- For **regression**, we use **Mean Squared Error (MSE)**:
  
$$
  \text{MSE} = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2
$$

- For **classification**, we use **Binary Cross-Entropy**:
  
$$
  \text{Loss} = -\frac{1}{m} \sum_{i=1}^{m} \left( y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right)
$$

Where \(p_i\) is the predicted probability of the positive class for example \(i\).

### 5. Optimization Algorithm

The model parameters \(\beta_0, \beta_1, \dots, \beta_n\) are adjusted to minimize the loss function. This process is called **optimization**.

The most common optimization algorithm is **Gradient Descent**, where we iteratively adjust the parameters in the direction of the negative gradient of the loss function:

$$
   \beta_j := \beta_j - \alpha \frac{\partial}{\partial \beta_j} \text{Loss}
$$

Where:
- \(\alpha\) is the **learning rate**, which controls the step size.

### 6. Gradient Calculation

In the case of **linear regression**, the gradient of the loss function with respect to a parameter \(\beta_j\) is given by:

$$
   \frac{\partial}{\partial \beta_j} \text{MSE} = -\frac{2}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i) x_{ij}
$$

Where \(x_{ij}\) is the value of the \(j\)-th feature for the \(i\)-th example.

### 7. Iterative Learning Process

The learning process follows these steps:

1. **Initialize Parameters**: Start with random values for \(\beta_0, \beta_1, \dots, \beta_n\).
2. **Compute Predictions**: Use the current parameters to compute the predicted values \(\hat{y}\).
3. **Calculate Loss**: Evaluate the loss function to measure how far the predictions are from the actual targets.
4. **Update Parameters**: Adjust the parameters using the gradient descent algorithm.
5. **Repeat**: Continue this process iteratively until the loss converges to a minimum or reaches an acceptable level.

These steps form the foundation of how machines learn from data. The goal is to minimize the error between predictions and actual outcomes by continually updating the model's parameters through optimization techniques like gradient descent.
# How Machines Learn

To understand how a machine can learn from the ground up, we can break it down into fundamental concepts and mathematical representations that illustrate the learning process.
### 1. Understanding Data and Features
- **Data Representation**: Data is typically represented as a matrix where each row is an example (or data point) and each column is a feature (or attribute) of that example.
$$
  X = \begin{bmatrix}
  x_{11} & x_{12} & \ldots & x_{1n} \\
  x_{21} & x_{22} & \ldots & x_{2n} \\
  \vdots & \vdots & \ddots & \vdots \\
  x_{m1} & x_{m2} & \ldots & x_{mn}
  \end{bmatrix}
$$
  where \($X$\) is the feature matrix, \($m$\) is the number of examples, and \($n$\) is the number of features.
  
### 2. Learning Objective
- **Function Approximation**: The primary goal is to find a function \($f$\) that maps input features \($X$\) to an output \($y$\). In supervised learning, this is typically a mapping of input to a target value (label).

$$
y = f(X) + \epsilon
$$
  
  where \(y\) is the target value, \(f(X)\) is the predicted value, and \(\epsilon\) is the error term.

### 3. Model Representation
- **Linear Model**: A simple model can be represented as a linear combination of the input features:
$$
\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_n x_n
$$
  where \($\hat{y}$\) is the predicted value, and \($\beta_0, \beta_1, \ldots, \beta_n$\) are the model parameters (coefficients).

### 4. Loss Function
- **Defining Error**: To learn from data, the model needs to measure how well it performs. This is done using a loss function that quantifies the difference between the predicted values and the actual target values.
  - **Mean Squared Error (MSE)** for regression:
$$
  \text{MSE} = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2
$$
  - **Binary Cross-Entropy** for classification:
$$
  \text{Loss} = -\frac{1}{m} \sum_{i=1}^{m} \left( y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right)
$$
  where \(p_i\) is the predicted probability of the positive class.

### 5. Optimization Algorithm
- **Adjusting Parameters**: The goal is to minimize the loss function by adjusting the model parameters. This is typically done using optimization algorithms like Gradient Descent.
- **Gradient Descent**: It iteratively updates the parameters in the opposite direction of the gradient of the loss function:
$$
   \beta_j := \beta_j - \alpha \frac{\partial}{\partial \beta_j} \text{Loss}
$$
  where \($\alpha$\) is the learning rate, controlling the step size.

### 6. Gradient Calculation
- **Calculating the Gradient**: For linear regression, the gradient of the loss with respect to \(\beta\) can be calculated as follows:
$$
   \frac{\partial}{\partial \beta_j} \text{MSE} = -\frac{2}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i) x_{ij}
$$
  where \($x_{ij}$\) is the feature value of the $j$-th feature for the $i$-th example.

### 7. Iterative Learning Process
1. **Initialize Parameters**: Start with random values for \(\beta\).
2. **Compute Predictions**: Use the current parameters to compute \(\hat{y}\).
3. **Calculate Loss**: Compute the loss using the loss function.
4. **Update Parameters**: Adjust the parameters using gradient descent.
5. **Repeat**: Iterate steps 2-4 until the loss converges or reaches an acceptable level.
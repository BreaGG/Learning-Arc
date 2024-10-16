# Neural Networks

### 1. What Are Neural Networks?
Neural networks are a class of machine learning models inspired by the structure and functioning of the human brain. They consist of interconnected nodes (neurons) organized into layers, enabling them to learn complex patterns and relationships in data.

### 2. Structure of a Neural Network
A typical neural network consists of three main types of layers:

- **Input Layer**: The first layer that receives input data. Each node in this layer corresponds to one feature of the input data.

- **Hidden Layers**: One or more layers between the input and output layers. Each hidden layer consists of nodes (neurons) that apply transformations to the input data using weights and activation functions.

- **Output Layer**: The final layer that produces the output of the network. In classification tasks, it usually has as many nodes as there are classes.

![[1 4PQOnabj78avPB2Mikk5GQ.jpg]]
### 3. Mathematical Representation
#### 3.1 Neuron Functioning
Each neuron receives input from the previous layer, processes it, and passes it to the next layer. The process can be mathematically represented as:
$$
z = \mathbf{w}^T \cdot \mathbf{x} + b
$$
where:
- ($z$) is the weighted sum of inputs,
- ($w$) is the weight vector,
- ($x$) is the input vector,
- ($b$) is the bias term.

#### 3.2 Activation Function
After calculating \( z \), an activation function is applied to introduce non-linearity:
$$
a = \phi(z)
$$
Common activation functions include:
- **Sigmoid**:
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$
- **ReLU (Rectified Linear Unit)**:
$$
\text{ReLU}(z) = \max(0, z)
$$
- **Tanh**:
$$
\tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}
$$

### 4. Forward Propagation
In forward propagation, the input data passes through the network layer by layer, and each neuron computes its output using the weighted sum and activation function. The process continues until the output layer produces the final prediction.

### 5. Loss Function
To assess how well the neural network performs, a loss function is used. The loss quantifies the difference between the predicted output and the actual target values:
- For regression tasks, Mean Squared Error (MSE) is commonly used:
$$
\text{MSE} = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2
$$
- For classification tasks, Cross-Entropy Loss is often used:
$$
\text{Loss} = -\frac{1}{m} \sum_{i=1}^{m} \sum_{c=1}^{C} y_{i,c} \log(p_{i,c})
$$
where ($C$) is the number of classes.

### 6. Backpropagation
To update the weights and biases of the network, backpropagation is used. It calculates the gradient of the loss function with respect to each weight by applying the chain rule of calculus. The update rule for weights is:
$$
w_j := w_j - \alpha \frac{\partial \text{Loss}}{\partial w_j}
$$
where ($\alpha$) is the learning rate.

### 7. Types of Neural Networks
- **Feedforward Neural Networks**: The simplest type where connections between nodes do not form cycles. Information moves in one direction—from input to output.

- **Convolutional Neural Networks (CNNs)**: Designed for image data, utilizing convolutional layers to automatically detect features like edges and textures.

- **Recurrent Neural Networks (RNNs)**: Suitable for sequential data (e.g., time series, text), with connections that allow information to persist over time, enabling the model to capture dependencies in sequences.
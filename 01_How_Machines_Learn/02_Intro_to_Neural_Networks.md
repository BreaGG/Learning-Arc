# Intro To Neural Network
> *A brief review after about neural networks.*
----
## Key Points on the History, Motivation, and Evolution of Deep Learning

> Learning Resource: *[Lecture: History, motivation, and evolution of Deep Learning](https://www.youtube.com/watch?v=0bMe_vCZo30&list=PLLHTzKZzVU9eaEyErdV26ikyolxOsz6mq&index=3&t=4939s)* by NYU, Yann LeCun

### The Roots of Deep Learning

Neural networks are inspired by the brain's function, but, like planes compared to birds, the inner mechanics are vastly different. In 1943, McCollough and Pitts introduced the concept of binary neurons (perceptrons) that could perform logical operations, akin to a Boolean circuit.

<div align="center" width=500>
<img src='https://www.researchgate.net/publication/356858632/figure/fig1/AS:1098794657693698@1638984469361/McCulloch-and-Pitts-Neuron-Model-13.ppm'/>
</div>
<br>

> *This makes me wonder: isn’t this conceptually similar to a Turing machine? With enough Boolean operations, could a network of binary neurons simulate any function that a Turing machine can?*

Later, Donald Hebb proposed the idea of synaptic plasticity (Hebbian learning), stating that neural connections could be altered through experience: *“Neurons that fire together, wire together.”*

Rosenblatt then advanced the field by building the first hardware implementation of the perceptron, known as the [*Mark I Perceptron Machine*](https://en.wikipedia.org/wiki/Perceptron). It utilized potentiometers as weights and electric motors to adjust them.

<div align='center'>
<img src='https://upload.wikimedia.org/wikipedia/en/thumb/5/52/Mark_I_perceptron.jpeg/220px-Mark_I_perceptron.jpeg'/><br>
<em style='font-size:12'>The Mark I Perceptron</em>
</div>
<br>

The initial excitement around perceptrons faded, however, as they failed to meet expectations for complex visual tasks. But the introduction of **backpropagation** brought them back into the spotlight.

It turned out that binary perceptrons were too simplistic due to their step function, which made their gradients unhelpful for learning. To enable backpropagation, a continuous and smooth activation function was needed.

- A step function like the perceptron produces no gradient (zero), rendering the derivative ineffective.
- At points of discontinuity, the gradient is undefined.

To address this, modern neural networks use neurons that compute a **weighted sum** via a linear combination. Earlier, this was computationally too expensive, as each activation required multiplying many neurons and inputs:

$$
n^m
$$

where \(n\) is the number of neurons and \(m\) is the number of inputs. The step function was cheaper computationally since it merely added a weight if a neuron was active.

With the advent of more powerful computers (thanks to Moore’s Law), we are now capable of deploying neural networks effectively.

### The Rise of Deep Learning Applications

**2009–2012**: Neural networks made strides in speech recognition.

**2012**: Convolutional Neural Networks (CNNs) revolutionized image classification.

**2015/2016**: Deep learning expanded into natural language processing (NLP).

### Early Pattern Recognition vs. Deep Learning

**Traditional Machine Learning** followed this flow:
$$
\text{Image} \rightarrow \text{Feature Extractor} \rightarrow \text{Trainable Classifier}
$$
The primary focus was on feature extraction: finding the right features to improve classification accuracy.

**Deep Learning** changes this by learning the task end-to-end:
$$
\text{Image} \rightarrow \text{Low-Level Features} \rightarrow \text{High-Level Features} \rightarrow \text{Trainable Classifier}
$$
Each layer is trainable and extracts patterns. The network becomes more powerful due to **non-linearity**, introduced by activation functions like ReLU:
$$
ReLU(x) = \max(0, x)
$$
This makes deep neural networks different from linear perceptrons, as non-linearity allows them to model more complex data.

### Gradient Descent and Backpropagation

For optimizing the model, we use **stochastic gradient descent** (SGD). Given a loss function \(L\), the gradient is computed, and the weights are updated using the rule:

$$
w_i = w_i - \alpha \frac{\partial L(W, X)}{\partial w_i}
$$

where \( \alpha \) is the learning rate and \( w_i \) is the weight at the \(i\)-th layer.

### Computing Gradients via Backpropagation

In a neural network, a hidden layer output might be represented as:

$$
h = ReLU(W_1 \mathbf{x} + \mathbf{b})
$$

Here, \(W_1\) are the weights for the first layer, \(\mathbf{x}\) is the input vector, and \(\mathbf{b}\) is the bias. For large input sizes (e.g., a 256x256 RGB image), the size of \( \mathbf{x} \) becomes \( 196,608 \), making fully connected weight matrices impractical.

Instead, techniques like **Convolutional Neural Networks (CNNs)** are used, which hypothesize the structure of the weight matrix to reduce computational cost.

### The Hierarchical Structure of the Brain and CNNs

The brain processes visual data through a feedforward mechanism similar to a CNN. Neurons in the retina compress the visual signal before it is processed by the visual cortex. Each stage abstracts features, from edges to objects, similar to CNN layers.

> *Interesting fact: humans have a blind spot in their retina.*

### Evolution of CNNs

Kunihiko Fukushima's early models for visual recognition laid the groundwork for LeCun's CNN architecture, LeNet.

<div align='center'>
<img src='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTWcZyKciourd227EsH5mT-wi7dty72houyBg&s'><br>
<em> LeCun's LeNet architecture</em>
</div>

These architectures use shared weights and kernels to perform convolutions over input data, followed by pooling. CNNs have since been used for tasks ranging from object detection to self-driving cars.

In 2012, AlexNet took the lead, with deeper networks like VGG (19 layers) and ResNet-50 (50 layers) following in its footsteps.

### Deep Learning: Feature Extraction

Images are compositional by nature, meaning they consist of parts, which consist of smaller details like edges, corners, and pixel values. Deep learning architectures allow neural networks to extract these features automatically through multiple layers.

### Manifold Learning Hypothesis

The **manifold hypothesis** suggests that high-dimensional data (e.g., images) actually lies on a lower-dimensional subspace (manifold) of the ambient space:

$$
\mathbb{R}^n \rightarrow \mathbb{R}^{\hat{n}} \text{ where } \hat{n} < n
$$

Manifold learning algorithms, like **t-SNE**, aim to reduce data dimensions while preserving the structure of the data manifold.

<div align="center">
<img width=400 src='https://www.researchgate.net/publication/341724327/figure/fig1/AS:896372609912832@1590723292607/Transformation-from-3-D-Samples-to-2-D-subspaces-Zhou-2016-Among-all-the-dimension.jpg'>
</div>

Deep learning allows us to discover these manifolds, extract features, and classify data more effectively.
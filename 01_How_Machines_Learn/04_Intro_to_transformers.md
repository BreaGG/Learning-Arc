# "Attention Is All You Need" – Understanding the Transformer Architecture

> *A deep dive into the Transformer model and how it changed the game in natural language processing.*

## Paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

The **Transformer** model, introduced by Vaswani et al. in 2017, revolutionized how we approach sequence modeling tasks like translation, language generation, and more. By eliminating the need for recurrent architectures (like RNNs and LSTMs), the Transformer made sequence modeling more efficient and scalable. The core idea behind this model is **self-attention**.

### Motivation: Why the Transformer?

Before Transformers, sequence models relied heavily on recurrence or convolution (e.g., LSTMs, GRUs, or CNNs). However, these models had issues with long-range dependencies, vanishing gradients, and limited parallelization due to their sequential nature.

The **Transformer** overcomes these challenges by using self-attention, which allows the model to focus on all parts of the input simultaneously, rather than processing it sequentially. This leads to faster training and better handling of long-term dependencies in sequences.

---

### Transformer Architecture Overview

The Transformer model is composed of two main parts:
1. **Encoder**: Processes the input sequence.
2. **Decoder**: Generates the output sequence, one token at a time, conditioned on the encoder's outputs.

Both the encoder and decoder are made up of several identical **layers**, each containing:
- **Multi-head self-attention mechanism**
- **Feedforward neural network**
- **Residual connections** followed by **layer normalization**

The architecture is designed to be highly parallelizable and more effective at capturing relationships between tokens over long distances, thanks to self-attention.

<div align="center">
    <img src="https://deeprevision.github.io/posts/001-transformer/transformer.png" alt="Transformer Architecture Overview" />
</div>

---

### Key Components of the Transformer

#### 1. Self-Attention Mechanism

The core innovation of the Transformer is the **self-attention** mechanism. Instead of processing tokens in sequence, the Transformer allows each token to “attend” to all other tokens in the input. This means that each token can directly consider the entire input sequence, making it easier to capture long-range dependencies.

Self-attention computes a weighted sum of the values, where the weights are calculated using the **queries (Q)**, **keys (K)**, and **values (V)**:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
$$

Where:
- \(Q\) = query matrix (derived from the input)
- \(K\) = key matrix (also derived from the input)
- \(V\) = value matrix (derived from the input)
- \(d_k\) = the dimensionality of the keys (used for scaling)

The result is a matrix that contains how much attention each word in the input sequence pays to every other word. The **softmax** function normalizes these weights so that they sum to 1.

<div align="center">
    <img src="https://lilianweng.github.io/posts/2018-06-24-attention/SAGAN.png" alt="Self-Attention Mechanism Diagram" />
</div>

#### 2. Multi-Head Attention

While the basic self-attention mechanism is powerful, the **Transformer** goes a step further with **multi-head attention**. Instead of using just one set of queries, keys, and values, it uses multiple sets (heads) in parallel. Each head focuses on different parts of the sequence, allowing the model to capture diverse types of relationships between tokens.

The output from each attention head is concatenated and linearly transformed:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W_O
$$

Where each attention head is calculated as:

$$
\text{head}_i = \text{Attention}(Q W_{iQ}, K W_{iK}, V W_{iV})
$$

This allows the model to learn more complex patterns in the data.

#### 3. Positional Encoding

One challenge with Transformers is that they don’t process the input sequence in order, like recurrent models do. This lack of order could prevent the model from knowing which tokens are first, next, or last. To solve this, the authors introduced **positional encodings**, which are added to the input embeddings to give the model a sense of the token positions in the sequence.

Positional encodings use sine and cosine functions at different frequencies:

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

Where \(pos\) is the position in the sequence and \(d_{\text{model}}\) is the dimensionality of the model. These encodings are added to the input embeddings, enabling the model to factor in token positions when attending to other tokens.

#### 4. Feedforward Networks

After self-attention, the output is passed through a **feedforward network**, which consists of two fully connected layers. This is applied independently to each position in the sequence:

$$
\text{FFN}(x) = \max(0, xW_1 + b_1) W_2 + b_2
$$

The feedforward network allows for further processing of the attention results, helping the model extract more complex patterns.

---

### Encoder-Decoder Structure

The **encoder** and **decoder** both use multiple layers of multi-head attention and feedforward networks. However, there’s a difference between them:

- The **encoder** only performs self-attention over the input sequence.
- The **decoder** performs two types of attention:
  - **Self-attention** over the decoder’s own inputs.
  - **Encoder-decoder attention**, which allows the decoder to focus on the relevant parts of the input sequence while generating the output.

The decoder generates the output sequence one token at a time, attending to both the input sequence and the previously generated tokens.

<div align="center">
    <img src="https://miro.medium.com/v2/resize:fit:731/1*MR6_IaOCoxbXRPuFbIJlUg.jpeg" alt="Encoder-Decoder Diagram" />
</div>

---

### Training and Objective

The model is trained using **teacher forcing**, where the true target token is passed into the decoder at each time step (instead of the predicted token). The objective is to minimize the difference between the predicted token probabilities and the true token probabilities, often using **cross-entropy loss**.

---

### Why the Transformer is Revolutionary

1. **Parallelization**: Unlike RNNs, where sequence processing is inherently sequential, Transformers allow for full parallelization, making them significantly faster to train.
2. **Long-Range Dependencies**: The self-attention mechanism enables the model to easily capture relationships between tokens regardless of their distance in the sequence.
3. **Scalability**: Transformers can be scaled to handle large datasets and long sequences without losing efficiency.
4. **Applicability**: Beyond NLP, Transformers have been applied to tasks in vision (Vision Transformers) and reinforcement learning, making them a general-purpose model for various domains.

---

### Key Contributions from the Paper

- The introduction of **self-attention** as a mechanism to replace recurrence and convolutions.
- The concept of **multi-head attention** to capture various relationships between tokens.
- **Positional encoding** to introduce the notion of order in sequences processed non-sequentially.
- Demonstrating that the Transformer can achieve state-of-the-art performance on translation tasks, such as the WMT English-to-German dataset.

---

### Final Thoughts

The **Transformer** is a groundbreaking architecture that has not only improved natural language processing tasks but has also inspired models in many other fields, like computer vision and reinforcement learning. By eliminating the need for recurrence and focusing on self-attention, the model is faster, more efficient, and capable of handling longer sequences better than its predecessors.

For more details, you can read the original paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

I also made my own approach and implementation of the paper: [Attention_Is_All_You_Need_From_Scratch](https://github.com/BreaGG/Attention_Is_All_You_Need_From_Scratch)
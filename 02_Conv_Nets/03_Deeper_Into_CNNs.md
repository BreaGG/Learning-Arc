# Deeper into Convolutional Neural Networks (CNNs)

CNNs use convolutions and pooling layers to process grid-like data, such as images. By sharing weights and exploiting local connections, CNNs are able to learn patterns in the data efficiently. In this section, we will look at the mathematical operations that define convolutions and explore how CNNs operate at a deeper level.

---

## 1. Cross-Correlation vs. Convolution

In practice, the operation typically referred to as "convolution" in CNNs is actually **cross-correlation**. The distinction is subtle but important in certain contexts. 

**Cross-correlation** applies a filter (kernel) to the input without flipping it. The output of cross-correlation can be written as:

$$
Z_{i,j,d} = u + \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} \sum_c W_{a,b,c,d} X_{i+a, j+b, c}
$$

Where:
- \(a, b\) are the indices of the kernel \(W\).
- \(c\) refers to the input channels (e.g., RGB channels in images).
- \(d\) is the output channel.
- \(X\) is the input data (image or feature map).
- \(\Delta\) represents the filter size.

### Convolution Operation

In a **true convolution**, the kernel is flipped before applying it to the input:

$$
Z_{i,j,d} = u + \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} \sum_c W_{-a, -b, c, d} X_{i+a, j+b, c}
$$

Notice the difference in indexing of the kernel weights \(W_{-a, -b}\) compared to cross-correlation. In many deep learning libraries, the cross-correlation operation is used and referred to as a "convolution."

### Why Flipping?

Flipping the kernel \(W\) makes the convolution operation **commutative**, which is useful in certain mathematical proofs and signal processing. However, in CNNs, flipping is generally not performed since the network will learn the correct patterns regardless of flipping.

---

## 2. Output Dimensions

The output dimensions of a convolutional layer depend on several factors:
- The size of the input.
- The size of the kernel (filter).
- The stride (how much the kernel moves).
- The padding (how much border is added to the input).

The output dimensions can be calculated as:

$$
Z_H = \frac{X_H + 2P - V_H}{S} + 1
$$

$$
Z_W = \frac{X_W + 2P - V_W}{S} + 1
$$

Where:
- \(Z_H\) and \(Z_W\) are the height and width of the output.
- \(X_H\) and \(X_W\) are the height and width of the input.
- \(V_H\) and \(V_W\) are the height and width of the kernel.
- \(P\) is the padding.
- \(S\) is the stride.

### Example:

If the input \(X\) is of size \(4 \times 4\) and the kernel \(V\) is of size \(2 \times 2\), with no stride or padding, the output dimensions are:

$$
Z_H = \frac{4 + 2(0) - 2}{1} + 1 = 3
$$

$$
Z_W = \frac{4 + 2(0) - 2}{1} + 1 = 3
$$

So the output will be a \(3 \times 3\) feature map.

With a stride of \(2\), the output becomes:

$$
Z_H = \frac{4 + 2(0) - 2}{2} + 1 = 2
$$

$$
Z_W = \frac{4 + 2(0) - 2}{2} + 1 = 2
$$

Now, the output is a \(2 \times 2\) feature map.

---

## 3. Padding and Stride

When applying convolutions, edge pixels are less frequently involved in the computation compared to central pixels. This can lead to a loss of important edge information. To mitigate this, **padding** is applied, which adds extra pixels (usually zeros) around the border of the input.

### Zero Padding:

Zero padding ensures that the output feature map has the same spatial dimensions as the input. For a kernel of size \(k_h \times k_w\), padding can be computed as:

$$
p_w = \frac{k_w - 1}{2}, \quad p_h = \frac{k_h - 1}{2}
$$

This is commonly referred to as **"same" padding**, meaning the input and output have the same height and width.

### Stride:

**Stride** controls how much the kernel moves at each step. A stride of 1 moves the kernel one pixel at a time, while a larger stride results in the kernel skipping pixels, effectively downsampling the input.

- A **larger stride** results in smaller output dimensions and reduced computational cost.
- A **smaller stride** retains more spatial detail at the expense of higher computational complexity.

---

## 4. Multiple Input and Output Channels

CNNs often process multi-channel inputs like RGB images (which have 3 color channels). For a multi-channel input, the kernel must also have a corresponding number of channels. The kernel has a depth equal to the number of input channels, and each slice of the kernel operates on a single channel of the input.

### Example of a 3-Channel Input:

Let’s say we have a \(3 \times 3\) image with 3 channels (RGB):

$$
X = \begin{bmatrix}
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, 
\begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}, 
\begin{bmatrix} 9 & 10 \\ 11 & 12 \end{bmatrix}
\end{bmatrix}
$$

A \(2 \times 2\) kernel with 3 channels would look like this:

$$
K = \begin{bmatrix}
\begin{bmatrix} -1 & -1 \\ 1 & 1 \end{bmatrix}, 
\begin{bmatrix} -2 & 2 \\ -2 & 2 \end{bmatrix}, 
\begin{bmatrix} -3 & 3 \\ -3 & 3 \end{bmatrix}
\end{bmatrix}
$$

The convolution of this input \(X\) with the kernel \(K\) across all channels produces an output feature map \(Z\).

### Multiple Output Channels

To increase the number of output channels, multiple sets of kernels are used. Each set of kernels produces its own output channel. For example, if you want 64 output channels, you will use 64 sets of kernels, each producing one output feature map.

---

## 5. Receptive Field

The **receptive field** of a neuron in a CNN is the region of the input that influences that neuron’s activation. As you move deeper into the network, the receptive field of each neuron increases, allowing the network to capture more global features.

For example, in the first layer, the receptive field may be a small \(3 \times 3\) patch, but by the time you reach deeper layers, the receptive field might cover a larger portion of the image.

The receptive field \(r_l\) at layer \(l\) can be computed as:

$$
r_l = 1 + \sum_{i=1}^l((k_i - 1) \prod_{k=1}^{i-1} s_k)
$$

Where:
- \(k_i\) is the kernel size at layer \(i\).
- \(s_k\) is the stride at layer \(k\).

A larger receptive field allows the network to capture more global patterns, while a smaller receptive field focuses on local patterns.

---

## 6. $1 \times 1$ Convolutions

A \(1 \times 1\) convolution applies a single scalar weight to each pixel, across all input channels, allowing the network to learn interactions between the channels without affecting the spatial resolution. This is particularly useful for reducing the number of channels or applying dimensionality reduction.

### Example:

Given an RGB image (3 channels), a \(1 \times 1\) convolution will combine the values of all three channels at each pixel into a single value:

$$
Z_{i,j} = R_{i,j} K_1 + G_{i,j} K_2 + B_{i,j} K_3
$$

Where \(R\), \(G\), and \(B\) are the red, green, and blue channels of the input, and \(K_1\), \(K_2\), and \(K_3\) are the corresponding weights in the kernel.

---

## 7. Pooling

Pooling layers are used to downsample the input, reducing the spatial dimensions while preserving important information. The two most common types are **max pooling** and **average pooling**.

- **Max Pooling**: Takes the maximum value

 from a region of the input.
  
  For a \(2 \times 2\) pooling region:

  $$
  \mathbf{P} = \max \begin{bmatrix}
  X_{11} & X_{12} \\
  X_{21} & X_{22}
  \end{bmatrix}
  $$

- **Average Pooling**: Takes the average value from a region.

Pooling introduces **translation invariance**, meaning small shifts in the input have little effect on the output.

---

## 8. Backpropagation in CNNs

Backpropagation in CNNs follows the same principles as in fully connected networks, using the **chain rule** to calculate gradients. However, since convolutional layers share weights (the same kernel is applied across different regions), the gradient for a specific weight is computed as the **sum of gradients** across all positions where the kernel was applied.

During backpropagation:
1. **Gradients with respect to the kernel** are computed by convolving the gradient of the loss with the input.
2. **Gradients with respect to the input** are computed using a transposed convolution with the flipped kernel.
3. **Gradients with respect to the bias** are the sum of the gradients over the entire feature map.

---

## Conclusion

CNNs are powerful because they take advantage of local spatial patterns through convolution and pooling layers. By stacking these layers, CNNs learn hierarchical features from the data, starting with simple patterns like edges and gradually building up to more complex representations like shapes and objects. With their ability to share weights and process multi-channel inputs, CNNs have become the go-to architecture for image-based tasks, among other applications.
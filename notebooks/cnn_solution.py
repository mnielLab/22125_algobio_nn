# FFN part
class SimpleFFNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases with small random values
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros(output_size)
    
    def relu(self, x):
        return np.maximum(0, x)

    # This version of sigmoid here is NOT numerically stable.
    # We need to split the cases where the input is positive or negative
    # because np.exp(-x) for something negative will quickly overflow if x is a large negative number
    # def sigmoid(self, x):
    #     return 1 / (1 + np.exp(-x))

    def sigmoid(self, x): 
        # This is equivalent to : 
        # if x>=0, then compute (1/(1+np.exp(-x)))
        # else: compute (np.exp(x)/(1+np.exp(x))))
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), 
                        np.exp(x) / (1 + np.exp(x)))

    def forward(self, x):
        """
        zi denotes the output of a hidden layer i
        ai denotes the output of an activation function at layer i
        (activations are relu, sigmoid, tanh, etc.)
        """

        # First layer
        z1 = np.dot(x, self.W1) + self.b1
        a1 = self.relu(z1)
        
        # Output layer
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.sigmoid(z2)
        
        # Return all the intermediate outputs as well because we need them for backpropagation (see slides)
        return z1, a1, z2, a2


class SimpleCNN:
    def __init__(self, kernel_size, in_channels, out_channels, hidden_size, output_size):
        self.filter = xavier_initialization_normal(kernel_size, in_channels, out_channels)
        self.fnn = SimpleFFNN(out_channels, hidden_size, output_size)

    def conv1d(self, batch_x):
        batch_size, input_length, in_channels = batch_x.shape
        kernel_size, _, out_channels = self.filter.shape
        output_length = input_length - kernel_size + 1
        batch_output = np.zeros((batch_size, output_length, out_channels))

        for i in range(output_length):
            for k in range(out_channels):
                for j in range(kernel_size):
                    # Extract the slice of input for the current position
                    input_slice = batch_x[:, i + j, :]
                    # Multiply element-wise and sum over in_channels
                    batch_output[:, i, k] += np.sum(input_slice * self.filter[j, :, k], axis=1)

        # Then use the numpy broadcasting to vectorize the operation instead
        # for i in range(output_size):
        #     batch_output[:, i, :] = np.sum(batch_x[:, i:i + kernel_size, :, None] * self.filter, axis=(1, 2))
    
        return batch_output

    def relu(self, x):
        return np.maximum(0, x)

    def global_max_pooling(self, x):
        # Find the maximum value for each feature map and its index
        pool_indices = np.argmax(x, axis=1)
        max_pool = np.max(x, axis=1)
        return max_pool, pool_indices

    def forward(self, x):
        # Pass the input through the convolution layer
        conv_output = self.conv1d(x)
        # Apply global max pooling
        pooled_output, pool_indices = self.global_max_pooling(conv_output)
        # Apply the ReLU activation function
        relu_output = self.relu(pooled_output)
        
        # Pass the result through the feedforward neural network
        return conv_output, pooled_output, pool_indices, relu_output, *self.fnn.forward(relu_output)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid_derivative(x):
    return x * (1 - x)

def backward(net, x, y, conv_output, pooled_output, pool_indices, relu_output, z1, a1, z2, a2, learning_rate=0.01):
    # Backwards pass for the FFNN
    error = a2 - y
    d_output = error * sigmoid_derivative(a2)

    d_W2 = np.dot(a1.T, d_output)
    d_b2 = np.sum(d_output, axis=0, keepdims=True)

    error_hidden_layer = np.dot(d_output, net.fnn.W2.T)
    d_hidden_layer = error_hidden_layer * relu_derivative(a1)

    d_W1 = np.dot(relu_output.T, d_hidden_layer)
    d_b1 = np.sum(d_hidden_layer, axis=0, keepdims=True)

    # Propagate the error back to the pooled layer
    d_pool = np.dot(d_hidden_layer, net.fnn.W1.T)

    # Loop through each sample in the batch
    batch_size, input_length, in_channels = x.shape
    kernel_size, in_channels, out_channels = net.filter.shape
    d_conv = np.zeros_like(conv_output)
    for b in range(batch_size):
        # Loop through each output channel
        for c in range(out_channels):
            # Get the index of the maximum value in the pooled output for this sample and channel
            max_index = pool_indices[b, c]
            # Propagate the gradient to the position before pooling and apply the ReLU derivative
            d_conv[b, max_index, c] = d_pool[b, c] * relu_derivative(pooled_output[b, c])
        
    # Then use this solution for the vectorized version
    # batch_indices = np.arange(len(x))[:, None]
    # channel_indices = np.arange(net.filter.shape[2])[None, :]
    # d_conv[batch_indices, pool_indices, channel_indices] = d_pool * relu_derivative(pooled_output)

    # Calculate the gradient for the convolution filter. First provide a solution for the explicit loop version
    # Loop through each sample in the batch
    d_filter = np.zeros_like(net.filter)
    for b in range(batch_size):
        # Loop through each position in the convolution output
        for i in range(x.shape[1] - d_filter.shape[0] + 1):
            # Extract the input slice corresponding to the current position
            input_slice = x[b, i:i + kernel_size, :]
            # Accumulate the gradient for the filter by multiplying the input slice with the corresponding gradient
            for j in range(kernel_size):
                for k in range(out_channels):
                    d_filter[j, :, k] += input_slice[j, :] * d_conv[b, i, k]


    # Then use this solution for the vectorized version
    # d_filter = np.zeros_like(net.filter)
    # for b in range(len(x)):
    #   for i in range(x.shape[1] - net.filter.shape[0] + 1):
    #       d_filter += x[b, i:i + net.filter.shape[0], :, None] * d_conv[b, i, None, :]

    # Update the filter weights and the weights and biases of the FFNN
    net.filter -= learning_rate * d_filter
    net.fnn.W1 -= learning_rate * d_W1
    net.fnn.b1 -= learning_rate * d_b1.squeeze()
    net.fnn.W2 -= learning_rate * d_W2
    net.fnn.b2 -= learning_rate * d_b2.squeeze()
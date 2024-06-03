import numpy as np
import pandas as pd
import math

import sys
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from ffnn_code import SimpleFFNN, relu_derivative, sigmoid_derivative

# Utility functions you will re-use

# Data-related utility functions
def load_blosum(filename):
    """
    Read in BLOSUM values into matrix.
    """
    aa = ['A', 'R', 'N' ,'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X']
    df = pd.read_csv(filename, sep='\s+', comment='#', index_col=0)
    return df.loc[aa, aa]

def load_peptide_target(filename):
    """
    Read amino acid sequence of peptides and
    corresponding log transformed IC50 binding values from text file.
    """
    df = pd.read_csv(filename, sep='\s+', usecols=[0,1], names=['peptide','target'])
    return df.sort_values(by='target', ascending=False).reset_index(drop=True)

def encode_peptides(X_in, blosum_file, max_pep_len=9):
    """
    Encode AA seq of peptides using BLOSUM50.
    Returns a tensor of encoded peptides of shape (1, max_pep_len, n_features) for a single batch
    """
    blosum = load_blosum(blosum_file)
    
    batch_size = len(X_in)
    n_features = len(blosum)
    
    X_out = np.zeros((batch_size, max_pep_len, n_features), dtype=np.int8)
    
    print(X_out.shape)

    for peptide_index, row in X_in.iterrows():
        for aa_index in range(len(row.peptide)):
            X_out[peptide_index, aa_index] = blosum[ row.peptide[aa_index] ].values
            
    return X_out, np.expand_dims(X_in.target.values,1)


# Misc. functions
def invoke(early_stopping, loss, model, implement=False):
    if implement == False:
        return False
    else:
        early_stopping(loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            return True
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SimpleCNN:
    def __init__(self, kernel_size, in_channels, out_channels, hidden_size, output_size):
        self.filter = np.random.randn(kernel_size, in_channels, out_channels)
        self.fnn = SimpleFFNN(out_channels, hidden_size, output_size)

    def conv1d(self, batch_x):
        """
        Function to perform 1D convolution on a batch of data.
        """
        batch_size = len(batch_x)
        kernel_size, in_channels, out_channels = self.filter.shape
        output_size = len(batch_x[0]) - kernel_size + 1
        batch_output = np.zeros((batch_size, output_size, out_channels))

        for b in range(batch_size):
            for i in range(output_size):
                for o in range(out_channels):
                    batch_output[b, i, o] = np.sum(batch_x[b, i:i + kernel_size] * self.filter[:, :, o])

        return batch_output

    def global_max_pooling(self, x):
        """
        Function to perform global max pooling and store max indices for backpropagation.
        """
        pool_indices = np.argmax(x, axis=1)
        return np.max(x, axis=1), pool_indices

    def forward(self, x):
        conv_output = self.conv1d(x)
        pooled_output, pool_indices = self.global_max_pooling(conv_output)
        return conv_output, pooled_output, pool_indices, *self.fnn.forward(pooled_output)


def backward(net, x, y, conv_output, pooled_output, pool_indices, z1, a1, z2, a2, learning_rate=0.01):
    """
    Function to backpropagate the gradients from the output to update the weights.
    """
    # Calculate loss gradient
    # Calculate loss gradient
    error = a2 - y
    d_output = error * sigmoid_derivative(a2)

    # Backpropagate to hidden layer
    d_W2 = np.dot(a1.T, d_output)
    d_b2 = np.sum(d_output, axis=0, keepdims=True)

    error_hidden_layer = np.dot(d_output, net.fnn.W2.T)
    d_hidden_layer = error_hidden_layer * relu_derivative(a1)

    # Backpropagate to input layer
    d_W1 = np.dot(pooled_output.T, d_hidden_layer)
    d_b1 = np.sum(d_hidden_layer, axis=0, keepdims=True)

    # Backpropagate to convolutional layer
    d_pool = np.dot(d_hidden_layer, net.fnn.W1.T)
    d_conv = np.zeros_like(conv_output)
    for b in range(len(x)):
        for o in range(net.filter.shape[2]):
            d_conv[b, pool_indices[b, o], o] = d_pool[b, o]
            
    # Gradient for the filter
    d_filter = np.zeros_like(net.filter)
    for b in range(len(x)):
        for i in range(d_conv.shape[1]):
            for o in range(net.filter.shape[2]):
                d_filter[:, :, o] += x[b, i:i + net.filter.shape[0]] * d_conv[b, i, o]

    # Update filter using gradient descent
    net.filter -= learning_rate * d_filter

    # Update filter using gradient descent
    net.fnn.W1 -= learning_rate * d_W1
    net.fnn.b1 -= learning_rate * d_b1.squeeze()
    net.fnn.W2 -= learning_rate * d_W2
    net.fnn.b2 -= learning_rate * d_b2.squeeze()
    net.filter -= learning_rate * d_filter


def train_network(net, x_train, y_train, learning_rate):
    """
    Trains the network for a single epoch, running the forward and backward pass, and compute and return the loss.
    """
    # Forward pass
    conv_output, pooled_output, pool_indices, z1, a1, z2, a2,  = net.forward(x_train)
    # backward pass
    backward(net, x_train, y_train, conv_output, pooled_output, pool_indices, z1, a1, z2, a2, learning_rate)
    loss = np.mean((a2 - y_train) ** 2)
    return loss
        
def eval_network(net, x_valid, y_valid):
    """
    Evaluates the network ; Note that we do not update weights (no backward pass)
    """
    conv_output, pooled_output, pool_indices, z1, a1, z2, a2, = net.forward(x_valid)
    loss = np.mean((a2-y_valid)**2)
    return loss

ALLELE = 'A0201' #'A0301'

blosum_file = './data/BLOSUM50'
train_data = f'./data/{ALLELE}/train_BA'
valid_data = f'./data/{ALLELE}/valid_BA'
test_data = f'./data/{ALLELE}/test_BA'

train_raw = load_peptide_target(train_data)
valid_raw = load_peptide_target(valid_data)
test_raw = load_peptide_target(test_data)

x_train_, y_train_ = encode_peptides(train_raw, blosum_file, train_raw.peptide.apply(len).max())
x_valid_, y_valid_ = encode_peptides(valid_raw, blosum_file, train_raw.peptide.apply(len).max())
x_test_, y_test_ = encode_peptides(test_raw, blosum_file, train_raw.peptide.apply(len).max())

print(x_train_.shape, x_valid_.shape, x_test_.shape)

# Using the full dataset as a batch (full gradient descent)
batch_size = x_train_.shape[0]
# The input size is the number of features ; Here it's max_length * 21 because we have 21 matrix dimensions
input_size = x_train_.shape[-1]

# Hyperparameters
learning_rate = float(sys.argv[1]) # 0.01
hidden_units = int(sys.argv[2]) # 50
n_epochs = int(sys.argv[3]) # 500
output_size = 1 # We want to predict a single value (regression)

# Neural Network training here
network = SimpleCNN(kernel_size=5, in_channels=21, out_channels=64, hidden_size=hidden_units, output_size=output_size)

train_losses = []
valid_losses = []
# add training part here 
for epoch in range(n_epochs):
    train_loss = train_network(network, x_train_, y_train_, learning_rate)
    valid_loss = eval_network(network, x_valid_, y_valid_)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    # For the first, every 5% of the epochs and last epoch, we print the loss 
    # to check that the model is properly training. (loss going down)
    if (n_epochs >= 10 and epoch % math.ceil(0.05 * n_epochs) == 0) or epoch == 0 or epoch == n_epochs:
        print(f"Epoch {epoch}: \n\tTrain Loss:{train_loss:.4f}\tValid Loss:{valid_loss:.4f}")

# Plotting the losses 
fig,ax = plt.subplots(1,1, figsize=(9,5))
ax.plot(range(n_epochs), train_losses, label='Train loss', c='b')
ax.plot(range(n_epochs), valid_losses, label='Valid loss', c='m')
ax.legend()
plt.show()






# CNN part
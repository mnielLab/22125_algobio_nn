import numpy as np
import pandas as pd
import math

import sys
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, matthews_corrcoef

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

ALLELE = 'A0301' #'A0301'

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

def relu_derivative(x):
    # to find
    return (x > 0).astype(float)

def sigmoid_derivative(x):
    # to find
    return x * (1 - x)

def backward(net, x, y, z1, a1, z2, a2, learning_rate=0.01):
    """
    Function to backpropagate the gradients from the output to update the weights.
    """
    # This assumes that we are computing a MSE as the loss function.
    # Look at your slides to compute the gradient backpropagation for a mean-squared error using the chain rule.

    # TO FIND 
    # Calculate loss gradient
    error = a2 - y
    d_output = error * sigmoid_derivative(a2)

    # TO FIND
    # Backpropagate to hidden layer
    d_W2 = np.dot(a1.T, d_output)
    d_b2 = np.sum(d_output, axis=0, keepdims=True)

    # TO FIND
    error_hidden_layer = np.dot(d_output, net.W2.T)
    d_hidden_layer = error_hidden_layer * relu_derivative(a1)

    # TO FIND
    # Backpropagate to input layer
    d_W1 = np.dot(x.T, d_hidden_layer)
    d_b1 = np.sum(d_hidden_layer, axis=0, keepdims=True)
    # TODO remove this print
    # print(d_W1, d_b1, d_W2, d_b2)
    # Update weights and biases using gradient descent
    net.W1 -= learning_rate * d_W1
    net.b1 -= learning_rate * d_b1.squeeze()
    net.W2 -= learning_rate * d_W2
    net.b2 -= learning_rate * d_b2.squeeze()

def train_network(net, x_train, y_train, learning_rate):
    """
    Trains the network for a single epoch, running the forward and backward pass, and compute and return the loss.
    """
    # Forward pass
    z1, a1, z2, a2,  = net.forward(x_train)
    # backward pass
    backward(net, x_train, y_train, z1, a1, z2, a2, learning_rate)
    loss = np.mean((a2 - y_train) ** 2)
    return loss
        
def eval_network(net, x_valid, y_valid):
    """
    Evaluates the network ; Note that we do not update weights (no backward pass)
    """
    z1, a1, z2, a2 = net.forward(x_valid)
    loss = np.mean((a2-y_valid)**2)
    return loss


# Reshaping the matrices so they're flat because feed-forward networks are "one-dimensional"
x_train_ = x_train_.reshape(x_train_.shape[0], -1)
x_valid_ = x_valid_.reshape(x_valid_.shape[0], -1)
x_test_ = x_test_.reshape(x_test_.shape[0], -1)

print(x_train_.shape, x_valid_.shape, x_test_.shape)

# Using the full dataset as a batch (full gradient descent)
batch_size = x_train_.shape[0]
# The input size is the number of features ; Here it's max_length * 21 because we have 21 matrix dimensions
input_size = x_train_.shape[1]

# CHECKPOINT

# Hyperparameters
learning_rate = float(sys.argv[1]) # 0.01
hidden_units = int(sys.argv[2]) # 50
n_epochs = int(sys.argv[3]) # 500
output_size = 1 # We want to predict a single value (regression)

# Neural Network training here
network = SimpleFFNN(input_size, hidden_units, output_size)

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
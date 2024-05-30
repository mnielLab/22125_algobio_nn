import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc, matthews_corrcoef

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
            
    return X_out, X_in.target.values

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

# FFN part
import numpy as np

class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases with small random values
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros(output_size)

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        # First layer
        z1 = np.dot(x, self.W1) + self.b1
        a1 = self.relu(z1)
        
        # Output layer
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.sigmoid(z2)
        
        # Return all the intermediate outputs as well because we need them for backpropagation (SEE YOUR SLIDES)
        return z1, a1, z2, a2

def relu_derivative(x):
    # to find
    return (x > 0).astype(float)

def sigmoid_derivative(x):
    # to find
    return x * (1 - x)

def backward(net, x, y, z1, a1, z2, a2, learning_rate=0.01):
    # TO FIND ENTIRE FUNCTION ish
    # Calculate loss gradient
    error = a2 - y
    d_output = error * sigmoid_derivative(output)

    # Backpropagate to hidden layer
    d_W2 = np.dot(a1.T, d_output)
    d_b2 = np.sum(d_output, axis=0, keepdims=True)

    error_hidden_layer = np.dot(d_output, net.W2.T)
    d_hidden_layer = error_hidden_layer * relu_derivative(a1)

    # Backpropagate to input layer
    d_W1 = np.dot(x.T, d_hidden_layer)
    d_b1 = np.sum(d_hidden_layer, axis=0, keepdims=True)

    # Update weights and biases using gradient descent
    net.W1 -= learning_rate * d_W1
    net.b1 -= learning_rate * d_b1.squeeze()
    net.W2 -= learning_rate * d_W2
    net.b2 -= learning_rate * d_b2.squeeze()

def train(net, x_train, y_train, epochs, learning_rate):
    for epoch in range(epochs):
        z1, a1, z2, a2,  = net.forward(x_train)
        backward(net, x_train, y_train, z1, a1, z2, a2, learning_rate)

        if epoch % 100 == 0:
            loss = np.mean((output - y_train) ** 2)
            print(f"Epoch {epoch}: Loss {loss}")



# Reshaping the matrices so they're flat 
x_train_ = x_train_.reshape(x_train_.shape[0], -1)
x_valid_ = x_valid_.reshape(x_valid_.shape[0], -1)
x_test_ = x_test_.reshape(x_test_.shape[0], -1)
batch_size = x_train_.shape[0]
n_features = x_train_.shape[1]

# CHECKPOINT

# Hyper parameters
learning_rate = 0.01
hidden_units = 10
n_epochs = 10
input_size = 9 * 21 # X * Y # Read your data to find out, where X is a length, and Y a matrix dimension
output_size = 1 # We want to predict a single value (regression)

# Neural Network training here
network = SimpleNN(input_size, hidden_size, output_size)

# add training part here 



# CNN part
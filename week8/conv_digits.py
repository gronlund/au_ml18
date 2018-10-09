import os
import urllib
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


# First we create a torch data loader - we have made that for you
def load_au_data(filename):
    """ Load and return the au digits data """
    if not os.path.exists(filename):
        print('file not exists - downloading')
        with open(filename, 'wb') as fh:
            path = "http://users-cs.au.dk/jallan/ml/data/{0}".format(filename)
            fh.write(urllib.request.urlopen(path).read())
    tmp = np.load(filename)
    au_digits = tmp['digits'] * 2 - 1
    au_labels = tmp['labels']
    print('data shape, type, min, max', au_digits.shape, au_digits.dtype, au_digits.min(), au_digits.max())
    print('labels shape and type', au_labels.shape, au_labels.dtype, au_labels.min(), au_labels.max())
    return au_digits, au_labels

    
def load_digits_train_data():
    """ load and return digits training data """    
    filename = 'auTrain.npz'
    return load_au_data(filename)


def load_digits_test_data():
    """ Load and return digits test data """
    filename = 'auTest.npz'
    return load_au_data(filename)
    


X_train, y_train = load_digits_train_data()
train_data = torch.utils.data.TensorDataset(torch.from_numpy(X_train.reshape(-1, 1, 28, 28)).float(), torch.from_numpy(y_train))
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

X_test, y_test = load_digits_test_data()
test_data = torch.utils.data.TensorDataset(torch.from_numpy(X_test.reshape(-1, 1, 28, 28)).float(), torch.from_numpy(y_test))
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

### YOUR CODE HERE
### END CODE
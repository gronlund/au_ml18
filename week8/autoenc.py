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
    

class DigitsDataset(Dataset):
    """ example of how to make a pytorch data set """
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float()

    
X_train, y_train = load_digits_train_data()
train_data = DigitsDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
# you can scan over train_loader to get data
### YOUR CODE HERE
### END CODE
print('Finished Training - lets plot some encodings')

# Assumes you have a class named net that supports forward to evaluate the neural net - if not fix the names etc. to make it work
fig, ax = plt.subplots(2, 8, figsize=(20, 16))
vis_loader = DataLoader(train_data, batch_size=1)
for i, timg in enumerate(vis_loader):
    with torch.no_grad():
        ax[0, i].imshow(timg.reshape(28, 28), cmap='gray')
        dec = net.forward(timg)
        ax[1, i].imshow(dec.reshape(28, 28), cmap='gray')
    if i >= 7: 
        print('break man')
        break

# Assumes you have a class named net has a linear layer named W1 - if not rename 
fig2, waxes = plt.subplots(4, 8, figsize=(20, 16))
with torch.no_grad():
    W1 = net.W1.weight.detach().numpy()

print('W1 shape', W1.shape)
for i, wax in enumerate(waxes.flat):
    w = W1[i, :]
    w = w/np.linalg.norm(w) # normalize
    wax.imshow(w.reshape(28, 28), cmap='gray')

plt.show()


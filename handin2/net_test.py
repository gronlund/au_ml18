import os
import numpy as np
import urllib
import urllib.request
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from net_classifier import NetClassifier, get_init_params

def export_fig(fig, name):
    result_path = os.path.join(os.getcwd(), 'results')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    print('outputting to file', name)
    my_path = os.path.join(result_path, name)
    fig.savefig(my_path)

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
    

def digits_test(hidden_size=256, epochs=50, batch_size=16, lr=0.1, reg=1e-4):
    net = NetClassifier()
    digits, labels = load_digits_train_data()
    digits_train, digits_val, labels_train, labels_val = train_test_split(digits, labels, test_size=0.15, random_state=42)
    init_params = get_init_params(digits.shape[1], hidden_size, 10)    
    net.fit(digits_train, labels_train, digits_val, labels_val, init_params, batch_size=batch_size, epochs=epochs, lr=lr, reg=reg)
    hist = net.history
    print('in sample accuracy', net.score(digits, labels))
    test_digits, test_labels = load_digits_test_data()
    print('test sample accuracy', net.score(test_digits, test_labels))
    fig, ax = plt.subplots(1, 2, figsize=(20, 16))  
    idx = list(range(hist['train_loss'].shape[0]))
    ax[0].plot(idx, hist['train_loss'], 'r-', linewidth=2, label='train loss')
    ax[0].plot(idx, hist['val_loss'], 'b-', linewidth=2, label='val loss')
    ax[0].set_title('Loss Per Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylim([0, 1])
    ax[1].plot(idx, hist['train_acc'], 'r-', linewidth=2, label='train acc')
    ax[1].plot(idx, hist['val_acc'], 'b-', linewidth=2, label='val acc')
    ax[1].set_title('Acccuracy Per Epoch')
    ax[1].set_ylim([0.5, 1])
    ax[1].set_ylabel('Accuracy')
    ax[1].set_xlabel('Epoch')
    plt.legend()
    export_fig(fig, 'epoch_plots.png')
    return net

if __name__ == '__main__':
    digits_test()

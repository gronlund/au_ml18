import os
import numpy as np
import urllib        
    
def load_digits_train_data():
    """ Load and return the training data """
    filename = 'auTrain.npz'
    if not os.path.exists(filename):
        print('file not exists - downloading')
        with open(filename, 'wb') as fh:
            path = "http://users-cs.au.dk/jallan/ml/data/{0}".format(filename)
            fh.write(urllib.request.urlopen(path).read())
    tmp = np.load('auTrain.npz')
    au_digits = tmp['digits']
    print('shape of input data', au_digits.shape)
    au_labels = np.squeeze(tmp['labels'])
    print('labels shape and type', au_labels.shape, au_labels.dtype)
    return au_digits, au_labels

def load_digits_test_data():
    """ Load and return the test data """
    filename = 'auTest.npz';
    if not os.path.exists(filename):
        print('file not exists - downloading')
        with open(filename, 'wb') as fh:
            path = "http://users-cs.au.dk/jallan/ml/data/{0}".format(filename)
            fh.write(urllib.request.urlopen(path).read())
    tmp = np.load('auTest.npz')
    au_digits = tmp['digits']
    print('shape of input data', au_digits.shape)
    au_labels = np.squeeze(tmp['labels'])
    print('labels shape and type', au_labels.shape, au_labels.dtype)
    return au_digits, au_labels

def print_score(classifier, X_train, X_test, y_train, y_test):
    """ Simple print score function that prints train and test score of classifier - almost not worth it"""
    print('In Sample Score: ',
          classifier.score(X_train, y_train))
    print('Test Score: ',
          classifier.score(X_test, y_test))

def export_fig(fig, name):
    result_path = os.path.join(os.getcwd(), 'results')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    my_path = os.path.join(result_path, name)
    fig.savefig(my_path)


def numerical_grad_check(f, x):
    """ Numerical Gradient Checker """
    eps = 1e-6
    h = 1e-4
    # d = x.shape[0]
    cost, grad = f(x)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        dim = it.multi_index
        tmp = x[dim]
        x[dim] = tmp + h
        cplus, _ = f(x)
        x[dim] = tmp - h 
        cminus, _ = f(x)
        x[dim] = tmp
        num_grad = (cplus-cminus)/(2*h)
        print('grad, num_grad, grad-num_grad', grad[dim], num_grad, grad[dim]-num_grad)
        assert np.abs(num_grad - grad[dim]) < eps, 'numerical gradient error index {0}, numerical gradient {1}, computed gradient {2}'.format(dim, num_grad, grad[dim])
        it.iternext()

    
            
            
    

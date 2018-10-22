from regression_stumps import RegressionStump
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from sklearn.datasets import load_boston
import pdb

class GradientBoostedRegressor():

    def __init__(self):
        # list of tuples/pairs (T_i, \alpha_i)
        self.model_list = []
        self.const_value = None
        
    def evaluate(self, X, model_list=None, const_value=None):
        """ Return the output of the model on input points
            This is const_value + sum_{T, a in model_list} a T(x)        

        Args:
            X: np.array shape n, d
            model_list: list of tuples (T, \alpha_i) - if empty use self.model_list
            const_value: scalar or None (the const model) - if none use self.const_value
        returns np.array, prediction on every point
        """
        # setup the right values
        if model_list is None:
            model_list = self.model_list
        if const_value is not None:
            f0 = const_value
        else:
            f0 = self.const_value
        pred = None
        ### YOUR CODE HERE
        ### END CODE
        return pred
    
    def cost_grad(self, X, y, model_list, const_value):
        """ Compute cost and gradient of model in model_list at each prediction point
            self.evaluate may come in handy

        Args:
            X: np.array shape n, d
            y: np.array shape n, 
            model_list: list of tuples (T, \alpha_i) or None
            const_value: scalar or None
        
        returns mean least absolute value cost and gradient at every point in X
        """
        cost = None
        grads = None
        ### YOUR CODE HERE
        ### END CODE
        return cost, grads

    def fix_leaf_values(self, X, y, model_list, const_value, rt):
        """ fix the leaf values of rt to minimize error 

        Args:
            X: np.array shape n, d
            y: np.array shape n, 
            model_list: list of tuples (T, \alpha_i) 
            const_value: scalar 
            rt: regression stump

        returns rt
        """
        ### YOUR CODE HERE
        ### END CODE
        return rt

    def fit(self, X, y, lr, rounds):
        """ Run Gradient Boosting Algorithm - remember to update to the negative gradient! 

        Args:
            X: np.array shape n, d
            y: np.array shape n, 
            lr: scalar learning rate
            rounds: int number of rounds to run gradient boosting
        
        sets model_list as list((T, lr))

        returns 
            hist: list of costs computed in each iteration
        """
        model_list = []
        hist = []
        ### YOUR CODE HERE
        ### END CODE
        # remember to set self.model_list and self.const_value
        return hist

    def predict(self, X):
        """ Make Model Prediction

        Args:
            X: np.array shape n, d

        returns 
            pred: np.array shape n,
        """
        
        pred = self.evaluate(X)
        return pred        

    def score(self, X, y):
        """ Compute mean absolute value cost

        Args:
            X: np.array shape n, d
            y: np.array shape n,

        returns 
            sc: scalar accuracy of model on data X with labels y
        """
        sc = None
        ### YOUR CODE HERE
        ### END CODE
        return sc



    
def main(rounds=10):
    boston = load_boston()
    # split 80/20 train-test
    X_train, X_test, y_train, y_test = train_test_split(boston.data,
                                                        boston.target,
                                                        test_size=0.15)

    baseline_accuracy_test = np.mean(np.abs(y_test - np.median(y_train)))
    baseline_accuracy_train = np.mean(np.abs(y_train - np.median(y_train)))
    print('Mean of Least Aboslute Cost of learning median of training data - train error:', baseline_accuracy_train) 
    print('Mean of Least Aboslute Cost of learning median of training data - test error:', baseline_accuracy_test) 
    print('Mean of Leaast Aboslute Cost of learning median of training data:', baseline_accuracy_test) 
    print('Lets see if we can do better with just one question')
    D = GradientBoostedRegressor()
    hist = D.fit(X_train, y_train, lr=0.1, rounds=rounds)
    print('Score of model', D.score(X_test, y_test))
    fig, ax = plt.subplots(1, 1)
    ax.plot(hist, 'b-.', linewidth=2, label='Least Absolute Cost')
    ax.set_title('Least Absolute Cost Per Iteration')
    ax.set_xlabel('Rounds')
    ax.set_ylabel('Mean Absolute Cost')
    plt.show()



if __name__ == '__main__':
    main(80)
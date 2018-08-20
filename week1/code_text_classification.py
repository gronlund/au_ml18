import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re
import os
import urllib

def standardize_strings(string_list):
    """ 
    Standardize strings by 
        1. making them lower case (example 'LaTeX' -> 'latex')
        2. remove any non-alphabetic characters, i.e. 
           characters not in [a-zæøå]. (example "l33t_h@x" -> "lthx")

    For example:
        >>> standardize_strings(['remove . please', 'also .,-_()'])
        ['remove  please', 'also ']
    
    The following methods might be useful: 
        str.lower,
        re.sub (regular expression from re package), 
        str.replace, 
        str.translate  (ALEX: documentation varies from Python 3.5 and 2.8, might be confusing).

    You are promised that the only characters in the input that are not [a-zæøå] 
    are one of the following:  .,-_() 

    You can ignore irrelevant spacing since we'll take care of it later. 

    Args:
    string_list: list of strings
    
    Returns:
    list of strings without non-alphabetic characters
    """
    res = []
    
    ### YOUR CODE 1-3 lines
    ### END CODE
    
    assert len(string_list) == len(res)
    return res

def split_strings(strings):
    """ 
    Split a list of strings into list of list of words. The splitting should be done on space ' '.

    For example: 
        >>> split_strings(['split me please', 'me to'])
        [['split', 'me', 'please'], ['me', 'to']]
        
    Try to use list comprehension instead of writing function with loops or recursion. 
    List comprehension works as follows

        >>> dat = list(range(6)) 
        >>> dat
        [0,1,2,3,4,5]

        >>> dat_squared = [x**2 for x in dat] 
        >>> dat_squared
        [0, 1, 4, 9, 16, 25]

    The following method might be useful:
        - str.split()  (for help with splitting strings)

    Args:
    strings: list of strings

    Returns:
    list of lists of strings (words)
    """
    word_lists = []
    
    ### YOUR CODE 
    ### END CODE
    
    assert len(word_lists) == len(strings)
    return word_lists

def remove_irrelevant_words(word_lists, bad_words=set()):
    """ 
    Remove the bad words from word_lists and returns the new list of list of words.
    The bad words are given by the argument 'bad_words'. 

    For example: 

        >>> words_list = [['remove', 'me', 'please'], ['also', 'you']]
        >>> bad_words = {'me', 'you'}
        >>> remove_irrelevant_words(words_list, bad_words)
        [['remove', 'please'], ['also']]

    Args:
    word_lists: list of list of string
    
    Returns: 
    list of list of strings (words) without words in bad_words
    """
    pruned_word_lists = []
    
    ### YOUR CODE, 1-3 lines
    ### END CODE
    
    assert len(pruned_word_lists) == len(word_lists)
    return pruned_word_lists

class TextClassifier():
    """
    Simple TextClassifier class (essentially a simple naive bayes class)
    """
    def  __init__(self):
        self.num_classes = None
        self.class_names = None
        self.class_probabilities = None
        self.prior = None
        self.strings = None
        self.labels = None
        self.vocabulary = None
        self.word_to_index = None
        self.index_to_word = None
        # takes from nltk.corpus.stopwords
        self.stopwords = {'over', 'hvad', 'man', 'sit', 'også', 'han', 'eller', 'blive', 'denne', 'det', 'meget', 'til',
                          'anden', 'hende', 'os', 'hvis', 'op', 'din', 'alt', 'havde', 'fra', 'en', 'af', 'efter', 'dig', 'mig', 'ind', 'hvor',
                          'den', 'alle', 'ikke', 'var', 'at', 'her', 'du', 'hos', 'jeg', 'der', 'ville', 'thi', 'ham', 'vil', 'under',
                          'ad', 'på', 'have', 'disse', 'for', 'skal', 'et', 'bliver', 'min', 'sig', 'i', 'nu', 'og', 'har', 'mine', 'vi',
                          'da', 'selv', 'jer', 'mit', 'sådan', 'hans', 'skulle', 'noget', 'som', 'vor', 'hendes', 'om', 'kunne', 'nogle',
                          'med', 'være', 'været', 'de', 'sine', 'men', 'dette', 'sin', 'deres', 'jo', 'ud', 'hun', 'er', 'ned', 'mange',
                          'mod', 'dog', 'når', 'end', 'blev', 'dem', 'formål', 'selskabets', 'virksomhed', 'samt'}
        
        
    def set_data(self, strings, labels, class_names=None):
        """
        Simple helper function
        """
        self.strings = strings # list of lists of strings
        self.labels = labels # numpy array integers in [0,1,...,num_classes]
        if class_names is None:
            self.class_names = np.unique(labels)
        else:
            self.class_names = class_names
        self.num_classes= np.unique(labels).size #
        '''print('The Classes: ', self.class_names)
        print('*'*10)
        print('The first 10 inputs - should be {0}'.format(self.class_names[0]))
        print(self.strings[0:10])
        print('*'*10)
        print('The Last 10 inputs - should be {0}'.format(self.class_names[1]))
        print(self.strings[-10:])'''

    def plot_words_histogram(self, n=20):
        """
        Plot histograms of the probabilities of the n most frequent words for each class

        We have already provided code for printing the top n words and frequencies.
        Your job is to make a bar plot of the frequencies 
        See 
        plt.figure, plt.bar, plt.plot, plt.show
        and
        See https://matplotlib.org/examples/api/barchart_demo.html
        
        Args:
        n: int number of words to plot
        
        Returns:
        """
        for c in range(self.num_classes):
            print('Class {0} - most probable words'.format(c))
            print('*'*20)
            probs = self.class_probabilities[c]
            n = np.min([n, len(probs)])
            best_indices = np.argsort(probs)[::-1][:n]
            best_freq = probs[best_indices]
            words = [self.index_to_word[x] for x in best_indices]
            for i, (w, f) in enumerate(zip(words, best_freq)):
                print('{0}: Word: {1} Probability {2}'.format(i+1, w, f))
            print('*' * 20)
            
            ### YOUR CODE approx. 5-10 lines 
            ### END CODE

    def make_vocabulary_and_index(self, word_lists):
        """
        Fill the vocabulary (set), word_to_index (dict), and index_to_word
        for the classifier.
         - self.vocabulary must be a set with all words in the list of lists of words word_lists
         - self.word_to_index is a dictionary that maps each word to a unique index (bijection)
         - self.index_to_word is a dictionary which is the inverse of word_to_index
        
        Use set and dict comprehension  (like list comprehension) to make each with one list of code
        {x for x in [1,2,3,4,5,6]} is a set comprehension  makes a set of elements in [1,2,3,4,5,6]
        
        As an example
        If word_lists is [['a', 'b'], ['c','b']] then vocabulary should be {'a', 'b', 'c'} and word_to_index should be something like {'a':0, 'b':1, 'c':2} (the permutation is irrelevant, this is just the obvious one.) 
        Also, index_to_word should be in this case {0:'a', 1:'b', 2:'c'}
        
        Args:
        word_lists: list of lists of strings
        
        Returns:
        """
        self.vocabulary = set()
        self.word_to_index = dict()
        self.index_to_word = dict()
        
        ### YOUR CODE 3-4 lines
        ### END CODE
        
        assert len(self.word_to_index) == len(self.index_to_word)

    def words_to_vectors(self, word_lists, vocabulary, index_map):
        """
        Each list of strings in word_lists is turned into a vector of length |vocabulary| in such a way that the ith entry of the vector is the number of occurences of word i (under index_map) in the string.
        index_map is the dictionary {word : index}
        
        As an example
        If the vocabulary is {'a','b','c','d','e'} and the mapping index_map is {'a':0,'b':1,'c':2,'d':3,'e':4}, then the word_list ['a','b','a'] is mapped to [2, 1, 0, 0, 0]

        A way of doing this is as follows (you can come up with a different implementation as long as the results are the same):
        1. Map each word to its index transforming list of words to list of indices
        2. Fill the vectors using these indices
        
        Step 1 can be done easily.
        For the second step the collections.Counter class may be useful: given a list it returns a dictionary-like object counting the ocurrences of each entry in the list. 
        As an example of this:
        c = Counter([1,1,2]) # -> Counter({1: 2, 2: 1})
        Now c.keys, c.values, c.items gives the list of indices and counts
        In [1]: list(c.keys())
        Out[1]: [1, 2]
        In [2]: list(c.values())
        Out[2]: [2, 1]
        
        Remember indexing in numpy:
        if x is a numpy array and ind a list of indices, then a[ind] indexes into the entries of x given in ind.

        Args:
        word_lists: list of lists of strings (each string is in self.vocabulary)
        
        Returns: 
        word_vectors: numpy array of size |word_lists| X |vocabulary|. The ith row is the vector to which the ith word_list is mapped by counting the occurrences of each word.
        """
        word_vectors = np.zeros((len(word_lists), len(vocabulary)))
        
        ### YOUR CODE
        ### END CODE
        
        return word_vectors

    def compute_class_word_probabilities(self, vectors):
        """
        Compute the probabilities of word w appearing in each class c
        Not all words occur in all classes which gives problems with zeros later.
        Therefore, for each class and word we compute the number of occurences plus one to estimate the probability of the word given the class.
        
        You must compute for each class c and word w
        p(w|c) = (1 + # occurences of w in string in class c) / (|vocabulary| + # words in class c)
        i.e. what is the probability of picking word w if we pick a random word from the strings of class c (with the 1 added to take care of zero issues).
        
        Furthermore, for each class c you must compute p(c), which is the "prior" probability of class c. 
        This is a probability distribution over classes given no other information (no words) and it is calculated as
        p(c) = (# number of strings of class c) / (# number of strings of all classes)
        
        The attribute self.num_classes is the total number of different classes
        The attribute self.labels is a list of length |strings| such that the ith entry (an integer in the range [0,...,#classes-1]) is the class of the ith string
        np.sum may be very handy here
        
        Args:
        vectors: numpy array of size (#strings) X |vocabulary|
        
        Returns: 
        - class_probs: numpy array of size self.num_classes X len(self.vocabulary)
        such that class_probs[c, w] is the probability of sampling word w from class c as defined above   
        - prior: numpy array of size num_classes such that prior[c] is the prior probability of class c.
        """
        n_input = vectors.shape[0]
        class_probs = np.zeros((self.num_classes, len(self.vocabulary)))
        prior = np.zeros(self.num_classes)
        
        ### YOUR CODE 6-10 lines
        ### END CODE
        
        return class_probs, prior

    def data_clean(self, strings, stopwords = set()):
        """
        Clean the data by calling the string functions you made
        Nothing required here
        """
        clean_data = standardize_strings(strings)
        word_lists = split_strings(clean_data)
        word_lists = remove_irrelevant_words(word_lists, stopwords)
        clean_word_lists = remove_irrelevant_words(word_lists)
        return clean_word_lists

    def train(self):
        """
        Train the text classifier
         - make strings to word lists
         - remove irrelevant words (a list is provided below) plus words occuring fewer than 5 times
         - make the word list to indicator numpy vectors and store in a large numpy array (bag of words)
         - for each class estimate the frequency of each word
         Just calling the functions you already made. 
         Nothing to do here.
        
        Returns:
        """
        assert self.strings is not None
        clean_word_lists = self.data_clean(self.strings, self.stopwords)
        self.make_vocabulary_and_index(clean_word_lists)
        string_vectors = self.words_to_vectors(clean_word_lists, self.vocabulary, self.word_to_index)
        self.class_probabilities, self.prior = self.compute_class_word_probabilities(string_vectors)

    def predict(self, strings):
        """
        Predict the correct label for the class of the strings input.
        We think that the right class is the one that more often uses the specific words in the input string.
        So, for each word in string see which classes often use this word.
        To compute the probability for all words multiply the frequencies.
        We will scale with the prior probability (i.e. if we had seen no words, what would we get) to handle cases with classes with very few data points.
        For each class compute the log of the product of frequencies of each word in
        string multiplied with prior for the class.

        In math terms for each class c compute
        prior(c) * \prod_{i}^{len(string)} ( p(string[i] | class c) )
        However, we're going to work in log transform space, so we actually compute the logarithm of this quantity:
        log(prior(c)) + \sum_{i}^{len(string)} log(p(string[i] | class c))

        Step 1: Clean Strings the same way training data was cleaned        
        Step 2: Compute the above probability for each class for each string (in log transform space)
                Here you must ignore words not seen in training data
        Step 3: Compute the argmax over these probabilities of each class for each string
        Step 4: Return the list (as numpy array) of the most probable class as the prediction for each input string

        functions np.sum, np.log may come very handy here.
        
        Args:
        strings: list of lists of strings
        
        Returns: 
        res: numpy array of length len(strings) where the ith entry is the label predicted class for the ith string in strings
        """
        if self.prior is None:
            print('Model not trained. Bad Dog. Train the model before using it for predictions!')
            return
        res = np.zeros(len(strings))
        cleaned_strings = self.data_clean(strings)
        for i, c in enumerate(cleaned_strings):            
            res[i] = 0 # dummy line you can remove when done , one line is needed 
            ### YOUR CODE  5-10 lines
            ### END CODE
            
            
        return res

# TEST CASES
# You can run the testcases as "python code_text_classification_solved.py -test"
# You should not waste time reading the test code. 

def test():
    test_string_cleaning()
    test_vectorize()
    test_class_word_probabilities()
    test_predict()
    print("ALL TEST CASES PASSED!")

def test_vectorize():
    """ 
    test words_to_vectors 
    """

    alphabet = 'abcdefghijklmnopqrstuvwxz'
    vocab = {x for x in alphabet}
    index_map = {x: i for (i, x) in enumerate(sorted(vocab))}
    a = [1, 2]
    b = [5]
    c = [0, 10]
    index_lists = [a, b, c]
    word_lists = [[alphabet[i] for i in x] for x in index_lists]
    tc = TextClassifier()

    print("TESTING 'make_vocabulary_and_index': \t\t", end='')
    tc.make_vocabulary_and_index(alphabet)

    assert vocab == tc.vocabulary, "Error with 'make_vocabulary_and_index'.\nExpected vocabulary: \t{0}\n Got: \t\t{1}".format(vocab, tc.vocabulary)

    # Test word_to_index and index_to_word by checking they are inverse mappings. 
    # The next two lines have errors, didn't have time to fix that. 
    # inv_map = {v: k for k, v in tc.word_to_index.items()} 
    # assert tc.index_to_word == inv_map, "Error with 'make_vocabulary_and_index'.\nExpected that the dictionaries 'index_to_word' and 'word_to_index' are each others inverse. This was not the case. "
    print("PASSED!")

    print("TESTING 'words_to_vector': \t\t\t", end='')
    string_vectors = tc.words_to_vectors(word_lists, vocabulary=vocab, index_map=index_map)
    target = np.zeros((3, len(vocab)))
    target[0, 1] = 1
    target[0, 2] = 1
    target[1, 5] = 1
    target[2, 0] = 1
    target[2, 10] = 1
    assert np.allclose(string_vectors, target)
    # print(string_vectors-target)
    print('PASSED!')




def test_class_word_probabilities():
    """ 
    Test class word probabilities  
    """
    print("TESTING 'compute_class_word_probabilities': \t", end='')
    # strings = ['i think machine learning is cool', 'he likes cake cake cake',
    #  'pandora has a box that should not be opened']
    strings = ['a a b', 'a', 'b b c c a a']
    labels = np.array([0, 0, 1])
    tc = TextClassifier()
    tc.set_data(strings, labels, ['dummy 0', 'dummy 1'])
    tc.train()
    assert tc.num_classes == 2
    assert tc.vocabulary == {'a', 'b', 'c'}
    assert np.allclose(tc.prior, np.array([2.0/3.0, 1.0/3.0]))
    #print('Class Probabilities of: ', end='')
    perm = [tc.word_to_index['a'], tc.word_to_index['b'], tc.word_to_index['c']]
    #print(tc.index_to_word[0], tc.index_to_word[1], tc.index_to_word[2])
    #print(perm)
    permuted = tc.class_probabilities[:, perm]
    #print('estimated:\n', permuted)
    true_class = np.array([[4/7, 2/7, 1/7], [3/9, 3/9, 3/9]])
    #print(true_class)
    assert np.allclose(permuted, true_class), 'expected to return {0}, got {1}'.format(true_class, permuted)
    #tc.plot_words_histogram()
    print('PASSED!')


def test_predict():
    """ 
    Test predict function in text classifier
    """
    print("TESTING 'predict':\t\t\t\t", end='')
    strings = ['a a a', 'b b', 'b b b c c c']
    labels = np.array([0, 0, 1])
    tc = TextClassifier()
    tc.set_data(strings, labels)
    tc.train()
    test_strings = ['a', 'b', 'c']
    predictions = tc.predict(test_strings)
    #print('Predicting: ', predictions)
    assert np.allclose(predictions, np.array([0, 0, 1]))
    print('PASSED!')

def test_string_cleaning():
    """ 
    Test string cleaning
    """
    strings = ['i THINK machine LEARNING is cool.,-()', 'he likes cake cake cake',
               'pandora has a box that should not be opened']

    bad_words = {'i', 'the', 'has', 'he', 'has', 'a', 'is'}
    result = [['think', 'machine', 'learning', 'cool'], ['likes', 'cake', 'cake', 'cake'],
              ['pandora', 'box', 'that', 'should', 'not', 'be', 'opened']]

    print("TESTING 'standardize_strings': \t\t\t", end='')
    expect = ['i think machine learning is cool', 'he likes cake cake cake', 'pandora has a box that should not be opened']
    strings_cleaned = standardize_strings(strings) # lower string remove special characters
    assert strings_cleaned == expect, "Error in 'standardize_strings'.\nExpected \t{0}\nGot \t\t{1}".format(expect, strings_cleaned)
    print('PASSED!')

    print("TESTING 'split_strings': \t\t\t", end='')
    expect = [['i', 'think', 'machine', 'learning', 'is', 'cool'], ['he', 'likes', 'cake', 'cake', 'cake'], ['pandora', 'has', 'a', 'box', 'that', 'should', 'not', 'be', 'opened']]
    word_lists = split_strings(strings_cleaned)
    assert word_lists == expect, "Error in 'split_strings': \nExpected\t{0}\nGot \t\t{1}".format(expect, word_lists)
    print("PASSED!")

    print("TESTING 'remove_irrelevant_words':\t\t", end='')
    clean_word_lists = remove_irrelevant_words(word_lists, bad_words)
    for right, wrong in zip(result, clean_word_lists):
        assert set(right) == set(wrong), "Error in 'remove_irrelevant_words'. \nExpected \t{0} \nGot:\t\t{1}".format(right, wrong)

    print("PASSED!")
    
    
def load_data():
    """ 
    Load the data in branche_data.npz and save it in object fields strings1-2 and labels (whose entries are in {0,1,..,num_classes-1})
    """
    filename = 'branchekoder_formal.gzip'
    
    if not os.path.exists(filename):
        with open(filename,'wb') as fh:
            fh.write(urllib.request.urlopen("https://users-cs.au.dk/jallan/ml/data/%s" % filename).read())
    #os.system('wget https://users-cs.au.dk/jallan/ml/data/{0}'.format(filename))

        
    data = pd.read_csv(filename, compression='gzip')
    strings1 = data[data.branchekode == 561010].formal.values
    strings2 = data[data.branchekode == 620100].formal.values
    labels = np.r_[np.zeros(len(strings1)), np.ones(len(strings2))]
    features = np.r_[strings1, strings2]    
    actual_class_names = ['Restauranter', 'Computerprogrammering']    
    return features, labels, actual_class_names

def split_data(features, labels):
    """ 
    Split data into train and test data 
    """
    size = labels.size
    split = int(size*3/4)
    # randomly permute data
    rp = np.random.permutation(size)
    feat = features[rp]
    lab = labels[rp]
    print(feat.shape)
    features_train = feat[:split]
    features_test = feat[split:]
    labels_train = lab[:split]
    labels_test = lab[split:]
    print('split data statistics: ')
    print('train data class 0: {0}'.format(100*(labels_train==0).mean()))
    print('train data class 1: {0}'.format(100*(labels_train==1).mean()))
    print('test data class 0: {0}'.format(100*(labels_test==0).mean()))
    print('test data class 1: {0}'.format(100*(labels_test==1).mean()))

    return features_train, labels_train, features_test, labels_test

def show_classifier_quality(predictions, true_labels, test_strings, labels):
    """ 
    Print accuracy and confusion matrix and some examples of errors 
    """
    print('Accuracy of the classifier: {0} %'.format(100*(predictions==true_labels).mean()))
    # mispredictions
    errors = predictions != true_labels
    # print('error is: ', errors.sum()/true_labels.size)
    first10 = errors.nonzero()[0][:10]
    print('First 1000 mispredictions')
    for s, p, t in zip(test_strings[first10], predictions[first10], true_labels[first10]):
        print('*'*20)
        print('String: {0}'.format(s))
        print('Predicted Class: {0} - {1}'.format(p, labels[p])) 
        print('Actual Class: {0} - {1}'.format(t, labels[int(t)]))       
        
def run():
    """
    Run on real data
    """
    print('Run on full data ')
    print('Load data')
    features, labels, names = load_data()
    features_train, labels_train, features_test, labels_test = split_data(features, labels)    
    tc = TextClassifier()    
    tc.set_data(features_train, labels_train, names)
    print('Train Classifier')
    tc.train()
    print('See Stats')
    tc.plot_words_histogram()
    print('Predict on test set')
    predictions_test = tc.predict(features_test).astype('int32')    
    show_classifier_quality(predictions_test, labels_test, features_test, tc.class_names)

if __name__ == '__main__':
    """ 
    Just add commands for each of the tests
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-test', action='store_true', default=False)
    parser.add_argument('-run', action='store_true', default=False)
    args = parser.parse_args()
    if args.test:
        test()
    if args.run:
        run()

# Selskabets formål er formueadministration     
# Selskabets formål er at udøve virksomhed med handel og service samt aktiviteter i tilknytning hertil. 

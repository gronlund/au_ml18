from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import os
import urllib
from sklearn.model_selection import train_test_split
from logistic_regression import LogisticRegressionClassifier
from h1_util import print_score

def load_branche_data(keys):
    """
    Load the data in branche_data.npz and save it in lists of
    strings and labels (whose entries are in {0,1,..,num_classes-1})
    """
    filename = 'branchekoder_formal.gzip'
    if not os.path.exists(filename):
        with open(filename, 'wb') as fh:
            path = "http://users-cs.au.dk/jallan/ml/data/{0}".format(filename)
            fh.write(urllib.request.urlopen(path).read())
    data = pd.read_csv(filename, compression='gzip')
    actual_class_names = []
    features = []
    labels = []
    for i, kv in enumerate(keys):
        key = kv[0]
        name = kv[1]
        strings = data[data.branchekode == key].formal.values
        features.extend(list(strings))
        label = [i] * len(strings)
        labels.extend(label)
        actual_class_names.append(name)
    assert len(features) == len(labels)
    features = np.array(features)
    labels = np.array(labels)
    return features, labels, actual_class_names


def get_branche_data(keys):
    features, labels, actual_class_names = load_branche_data(keys)
    X_train, X_test, y_train, y_test = train_test_split(features, labels)
    return X_train, X_test, y_train, y_test, actual_class_names


def print_errors(classifier, feat, raw_feat, labels, class_names, top=10):
    """ Print first top errors made """
    pred = classifier.predict(feat)
    idx = pred != labels
    top_errors = np.nonzero(idx)
    top_errors = top_errors[0]
    top_errors = top_errors[0:top]
    print('*'*30)
    for err_idx in top_errors:        
        print('\nMispredicted: ', raw_feat[err_idx])
        print('Classifier Prediction: ', pred[err_idx], class_names[pred[err_idx]])
        print('Actual Label: ', labels[err_idx], class_names[labels[err_idx]])
    print('*'*30)


def branche_data_test():
    keys = [(561010, 'Restauranter'), (620100, 'Computerprogrammering')]
    feat_train, feat_test, y_train, y_test, cnames = get_branche_data(keys)
    c = CountVectorizer()
    c.fit(feat_train)
    bag_of_words_feat_train = c.transform(feat_train).toarray()
    classifier = LogisticRegressionClassifier()
    classifier.fit(bag_of_words_feat_train, y_train, lr=0.1, batch_size=16, epochs=50)
    print('Logistic Regression Industri Codes Classifier')
    bag_of_words_feat_test = c.transform(feat_test).toarray()
    print_score(classifier, bag_of_words_feat_train, bag_of_words_feat_test, y_train, y_test)


if __name__=='__main__':
    branche_data_test()

import os
import pandas as pd
# from pprint import pprint
import numpy as np

import matplotlib.pyplot as plt

from models.utils import *
from models.svm import * # svm_grid, experiment_train, experiment_intent

from sklearn import svm
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

df_test = pd.read_pickle('../data/Marie/Marie_test.pickle') # Braun
df_train = pd.read_pickle('../data/Marie/Marie_train.pickle') # Braun

X_test = df_test['utterance']
y_test = df_test['intent']

X = df_train['utterance']
y = df_train['intent']

texts_train = remove_stopwords(remove_non_alpha_num(X))
texts_test = remove_stopwords(remove_non_alpha_num(X_test))


# Load Facebook's pretrained FastText embedding
from gensim.models import KeyedVectors
# Load Facebooks's pre-trained FastText embeddings
# https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md
word_vectors = KeyedVectors.load_word2vec_format('C:\Local_data\wiki.en\wiki.en.vec', binary=False)

# Load Stanford's pre-trained GloVe embeddings
from gensim.scripts.glove2word2vec import glove2word2vec
glove2word2vec(glove_input_file='C:\Local_data\glove.6B\glove.6B.300d.txt', word2vec_output_file='C:\Local_data\glove.6B\gensim_glove.6B.300d.txt')
word_vectors = KeyedVectors.load_word2vec_format('C:\Local_data\glove.6B\gensim_glove.6B.300d.txt', binary=False)

# Load Google's pre-trained Word2Vec model
#model = gensim.models.Word2Vec.load_word2vec_format(, binary=True)
word_vectors = KeyedVectors.load_word2vec_format('C:\Local_data\GoogleNews-vectors\GoogleNews-vectors-negative300.bin', binary=True)




train_sum, train_ave = create_input(texts_train, word_vectors)
test_sum, test_ave = create_input(texts_test, word_vectors)



# def svm_grid(X, y, X_test, y_test, class_weight='balanced'):

#     param_grid = [
#       {'C': [0.01, 0.1, 1, 10, 100, 1000], 'kernel': ['linear']},
#       {'C': [10, 100, 1000, 10000, 100000], 'gamma': [0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001], 'kernel': ['rbf']},
#      ]

#     clf = svm.SVC(class_weight=class_weight)

#     grid = GridSearchCV(clf, param_grid=param_grid, scoring='f1_macro', cv=4)
#     paramsearch = grid.fit(X,y)

#     param_cv_results = pd.DataFrame(paramsearch.cv_results_) # return this

#     bestSVM = paramsearch.best_estimator_

#     svm_model = bestSVM.fit(X,y)
#     y_hat = svm_model.predict(X_test)

#     prec_macro = precision_score(y_test, y_hat, average='macro')
#     rec_macro = recall_score(y_test,y_hat, average='macro')
#     f1_macro = f1_score(y_test,y_hat, average= 'macro')

#     confusion = pd.DataFrame(confusion_matrix(y_test, y_hat)) # return this

#     y_hat_train = svm_model.predict(X)

#     prec_macro_train = precision_score(y, y_hat_train, average='macro')
#     rec_macro_train = recall_score(y,y_hat_train, average='macro')
#     f1_macro_train = f1_score(y,y_hat_train, average= 'macro')

#     return [rec_macro,rec_macro_train], [prec_macro,prec_macro_train], [f1_macro,f1_macro_train], param_cv_results, confusion

# Full data
rec_macro, prec_macro, f1_macro, param_cv_results, confusion = svm_grid(train_sum, y, test_sum, y_test, class_weight='balanced')

results_train = pd.DataFrame(dict(
                    percent=[100],
                    test=[True],
                    recall=[rec_macro[0]],
                    precision=[prec_macro[0]],
                    F1=[f1_macro[0]]))

# training score
results_train = results_train.append(dict(
                    percent=100,
                    test=False,
                    recall=rec_macro[1],
                    precision=prec_macro[1],
                    F1=f1_macro[1]), ignore_index=True)

# varying trainin observations
for percent in [80,60,40,20]:
    results_train = results_train.append(experiment_train(percent, word_vectors))

results_train['percent'] = pd.to_numeric(results_train['percent'])
results_train[results_train['test']==True].plot.scatter(x='percent',y='F1', ylim=(0,1))
results_train[results_train['test']==False].plot.scatter(x='percent',y='F1', ylim=(0,1))
results_train['model'] = 'SVM FastText cw' # SVM GloVe cw
results_train.to_pickle('../results/results_train_svm_ft_cw') # gv

# varying intents
# def intents_experiment(j):
#     for i in range(10):
#         fn_train = DIR +'df_train_' + str(j) + '_' + str(i)
#         df_train = pd.read_pickle(fn_train)

#     #         fn_train_val = DIR +'df_train_val_' + str(j) + '_' + str(i)
#     #         df_train_val = pd.read_pickle(fn_train_val)

#     #         fn_val = DIR +'df_val_' + str(j) + '_' + str(i)
#     #         df_val = pd.read_pickle(fn_val)

#         fn_test = DIR +'df_train_' + str(j) + '_' + str(i)
#         df_test = pd.read_pickle(fn_test)

#         X = df_train['utterance']
#         y = df_train['intent']

#         X_test = df_test['utterance']
#         y_test = df_test['intent']

#         texts_train = remove_stopwords(remove_non_alpha_num(X))
#         texts_test = remove_stopwords(remove_non_alpha_num(X_test))

#         train_sum, train_ave = create_input(texts_train, word_vectors)
#         test_sum, test_ave = create_input(texts_test, word_vectors)

#         rec_macro, prec_macro, f1_macro, _, _ = svm_grid(train_sum, y, test_sum, y_test, class_weight='balanced')

#         results_intent = results_intent.append(dict(
#                     intents=int(j),
#                     test=True,
#                     recall=rec_macro[0],
#                     precision=prec_macro[0],
#                     F1=f1_macro[0]), ignore_index=True)

#         # training score
#         results_intent = results_intent.append(dict(
#                     intents = int(j),
#                     test=False,
#                     recall=rec_macro[1],
#                     precision=prec_macro[1],
#                     F1=f1_macro[1]), ignore_index=True)

DIR = '../Data/Marie/intent/'
# global results_intent
results_intent = pd.DataFrame(columns=['intents','test','recall','precision','F1'])

results = []

for n_intent in [5,10,15,20,25,30,35,40,45,50]:
    results_intent = results_intent.append(experiment_intent(n_intent, word_vectors, DIR))

results_intent['intents'] = pd.to_numeric(results_intent['intents'])
results_intent[results_intent['test']==True].plot.scatter(x='intents',y='F1', ylim=(0,1))
results_intent[results_intent['test']==False].plot.scatter(x='intents',y='F1', ylim=(0,1))
results_intent['model'] = 'SVM FastText cw' # SVM GloVe cw
results_intent.to_pickle('../results/results_intent_svm_ft_cw') # gv

###################################################################

# Braun Original

DIR = '../data/Braun/original/'
class_weight='balanced'

for data in ['travel','ubuntu','webapp']:
    fn_test = DIR + data + '_test.pickle'
    df_test = pd.read_pickle(fn_test)
    
    fn_train = DIR + data + '_train.pickle'
    df_train = pd.read_pickle(fn_train)

    X_test = df_test['utterance']
    y_test = df_test['intent']

    X = df_train['utterance']
    y = df_train['intent']

    texts_train = remove_stopwords(remove_non_alpha_num(X))
    texts_test = remove_stopwords(remove_non_alpha_num(X_test))

    train_sum, train_ave = create_input(texts_train, word_vectors)
    test_sum, test_ave = create_input(texts_test, word_vectors)
    
    rec_macro, prec_macro, f1_macro, param_cv_results, confusion = svm_grid(train_sum, y, test_sum, y_test, class_weight='balanced')

    results_train = pd.DataFrame(dict(
                        percent=[100],
                        test=[True],
                        recall=[rec_macro[0]],
                        precision=[prec_macro[0]],
                        F1=[f1_macro[0]]))

    # training score
    results_train = results_train.append(dict(
                        percent=100,
                        test=False,
                        recall=rec_macro[1],
                        precision=prec_macro[1],
                        F1=f1_macro[1]), ignore_index=True)
    
    results_train['model'] = 'SVM FastText'
    
    results_train.to_pickle('../results/braun/results_'+ data + '_svm_ft.pickle')

###################################################################
from multiprocessing import Pool
global DIR 
DIR = '../Data/Marie/intent/'
global results_intent 
results_intent = pd.DataFrame(columns=['intents','test','recall','precision','F1'])
from multiprocessing import Pool

global output
output =[]

def f(x):
    y = x**2
    output.append(y)

if __name__ == '__main__':
    with Pool(7) as p:
        p.map(f, [5,10,15,20,25,30,35,40,45,50])
#####################################################################

# TF_IDF

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

from sklearn.feature_extraction.text import TfidfVectorizer
v = TfidfVectorizer(stop_words='english') # preprocessor, strip_accents, ngram_range, 

df_test = pd.read_pickle('../data/Marie_test.pickle')
df_train = pd.read_pickle('../data/Marie_train.pickle')
df_train_80 = pd.read_pickle('../data/Marie_train_80.pickle')
df_train_60 = pd.read_pickle('../data/Marie_train_60.pickle')
df_train_40 = pd.read_pickle('../data/Marie_train_40.pickle')
df_train_20 = pd.read_pickle('../data/Marie_train_20.pickle')

dfs = [df_train, df_train_80, df_train_60, df_train_40, df_train_20]

def stem(df_in):
    df_in['stemmed'] = df_in.utterance.map(lambda x: ' '.join([stemmer.stem(y) for y in x.split(' ')]))
    return df_in

df_train = stem(df_train)
df_test = stem(df_test)
dfs = [stem(df) for df in dfs]

list_array = []
list_array_test = []
for i in range(len(dfs)):
    df = dfs[i]
    list_array.append(v.fit_transform(df['stemmed']).toarray())
    list_array_test.append(v.transform(df_test['stemmed']).toarray())

    
dfs_out = [pd.DataFrame(array) for array in list_array]
dfs_test = [pd.DataFrame(array) for array in list_array_test]

X = dfs_out[0]
y = df_train['intent']

X_80 = dfs_out[1]
y_80 = df_train_80['intent']

X_60 = dfs_out[2]
y_60 = df_train_60['intent']

X_40 = dfs_out[3]
y_40 = df_train_40['intent']

X_20 = dfs_out[4]
y_20 = df_train_20['intent']

# X_test = dfs_out[5]
y_test = df_test['intent']

from sklearn import svm
from sklearn.model_selection import GridSearchCV

param_grid = [
  {'C': [0.01, 0.1, 1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [10, 100, 1000, 10000, 100000], 'gamma': [0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001], 'kernel': ['rbf']},
 ]

clf = svm.SVC()

grid = GridSearchCV(clf, param_grid=param_grid, scoring='f1_macro', cv=4)
paramsearch = grid.fit(X,y)

pd.DataFrame(paramsearch.cv_results_)

bestSVM = paramsearch.best_estimator_

#clf = svm.SVC(kernel='linear',C=1, random_state=None)
svm = bestSVM.fit(X,y)
y_hat = svm.predict(dfs_test[0])
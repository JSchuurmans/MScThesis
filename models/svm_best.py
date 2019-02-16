from sklearn import svm
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# from models.utils import *

import pandas as pd

def svm_best(X, y, class_weight='balanced'):
    param_grid = [
        {'C': [0.01, 0.1, 1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [10, 100, 1000, 10000, 100000], 'gamma': [0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001], 'kernel': ['rbf']}
        ]

    clf = svm.SVC(class_weight=class_weight)
    grid = GridSearchCV(clf, param_grid=param_grid, scoring='f1_macro', cv=4)
    paramsearch = grid.fit(X,y)
    param_cv_results = pd.DataFrame(paramsearch.cv_results_) # return this
    bestSVM = paramsearch.best_estimator_
    return bestSVM, param_cv_results


	# svm_model = bestSVM.fit(X,y)
	# y_hat = svm_model.predict(X_test)

	# prec_macro = precision_score(y_test, y_hat, average='macro')
	# rec_macro = recall_score(y_test,y_hat, average='macro')
	# f1_macro = f1_score(y_test,y_hat, average= 'macro')

	# confusion = pd.DataFrame(confusion_matrix(y_test, y_hat)) # return this

	# y_hat_train = svm_model.predict(X)

	# prec_macro_train = precision_score(y, y_hat_train, average='macro')
	# rec_macro_train = recall_score(y,y_hat_train, average='macro')
	# f1_macro_train = f1_score(y,y_hat_train, average= 'macro')

	# return [rec_macro,rec_macro_train], [prec_macro,prec_macro_train], [f1_macro,f1_macro_train], param_cv_results, confusion
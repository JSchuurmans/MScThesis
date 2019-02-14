from sklearn import svm
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# from models.utils import *

import pandas as pd

def svm_grid(X, y, X_test, y_test, class_weight='balanced'):

	param_grid = [
	  {'C': [0.01, 0.1, 1, 10, 100, 1000], 'kernel': ['linear']},
	  {'C': [10, 100, 1000, 10000, 100000], 'gamma': [0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001], 'kernel': ['rbf']},
	 ]

	clf = svm.SVC(class_weight=class_weight)

	grid = GridSearchCV(clf, param_grid=param_grid, scoring='f1_macro', cv=4)
	paramsearch = grid.fit(X,y)

	param_cv_results = pd.DataFrame(paramsearch.cv_results_) # return this

	bestSVM = paramsearch.best_estimator_

	svm_model = bestSVM.fit(X,y)
	y_hat = svm_model.predict(X_test)

	prec_macro = precision_score(y_test, y_hat, average='macro')
	rec_macro = recall_score(y_test,y_hat, average='macro')
	f1_macro = f1_score(y_test,y_hat, average= 'macro')

	confusion = pd.DataFrame(confusion_matrix(y_test, y_hat)) # return this

	y_hat_train = svm_model.predict(X)

	prec_macro_train = precision_score(y, y_hat_train, average='macro')
	rec_macro_train = recall_score(y,y_hat_train, average='macro')
	f1_macro_train = f1_score(y,y_hat_train, average= 'macro')

	return [rec_macro,rec_macro_train], [prec_macro,prec_macro_train], [f1_macro,f1_macro_train], param_cv_results, confusion

def experiment_train(percent, word_vectors, class_weight='balanced'):
	# global results_train

	results_train = pd.DataFrame(columns=['percent','test','recall','precision','F1'])
	DIR = '../Data/Marie/train_obs/'
	fn_test = '../data/Marie/Marie_test.pickle'

	for i in range(10):
		fn = DIR + 'df_train_' + str(percent) +'_' + str(i)
		df = pd.read_pickle(fn)
		X = df['utterance']
		y = df['intent']

		# fn_test = DIR + 'df_test_' + str(percent) +'_' + str(i)
		df_test = pd.read_pickle(fn_test)
		X_test = df_test['utterance']
		y_test = df_test['intent']

		texts_train = remove_stopwords(remove_non_alpha_num(X))
		texts_test = remove_stopwords(remove_non_alpha_num(X_test))

		train_sum, train_ave = create_input(texts_train, word_vectors)
		test_sum, test_ave = create_input(texts_test, word_vectors)

		rec_macro, prec_macro, f1_macro, _, _ = svm_grid(train_sum, y, test_sum, y_test, class_weight=class_weight)

		results_train = results_train.append(dict(
			percent=int(percent),
			test=True,
			recall=rec_macro[0],
			precision=prec_macro[0],
			F1=f1_macro[0]), ignore_index=True)

		# training score        
		results_train = results_train.append(dict(
			percent=int(percent),
			test=False,
			recall=rec_macro[1],
			precision=prec_macro[1],
			F1=f1_macro[1]), ignore_index=True)
	
	return results_train



def experiment_intent(n_intent, word_vectors, DIR, class_weight='balanced', set='df'):
	# global results_intent

	results_intent = pd.DataFrame(columns=['intents','test','recall','precision','F1'])

	for i in range(10):
		fn_train = DIR + set +'_train_' + str(n_intent) + '_' + str(i)
		df_train = pd.read_pickle(fn_train)
#         fn_train_val = DIR +'df_train_val_' + str(j) + '_' + str(i)
#         df_train_val = pd.read_pickle(fn_train_val)
        
#         fn_val = DIR +'df_val_' + str(j) + '_' + str(i)
#         df_val = pd.read_pickle(fn_val)
		fn_test = DIR + set + '_test_' + str(n_intent) + '_' + str(i)
		df_test = pd.read_pickle(fn_test)

		X = df_train['utterance']
		y = df_train['intent']

		X_test = df_test['utterance']
		y_test = df_test['intent']

		texts_train = remove_stopwords(remove_non_alpha_num(X))
		texts_test = remove_stopwords(remove_non_alpha_num(X_test))

		train_sum, train_ave = create_input(texts_train, word_vectors)
		test_sum, test_ave = create_input(texts_test, word_vectors)

		rec_macro, prec_macro, f1_macro, _, _ = svm_grid(train_sum, y, test_sum, y_test, class_weight=class_weight)

		results_intent = results_intent.append(dict(
			intents=int(n_intent),
			test=True,
			recall=rec_macro[0],
			precision=prec_macro[0],
			F1=f1_macro[0]), ignore_index=True)

        # training score
		results_intent = results_intent.append(dict(
			intents = int(n_intent),
			test=False,
			recall=rec_macro[1],
			precision=prec_macro[1],
			F1=f1_macro[1]), ignore_index=True)

	return results_intent
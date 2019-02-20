import os
import pandas as pd
import numpy as np

from models.naive_bayes import nb

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_validate

###############################################
# RETAIL
###############################################

### Full data ###################################
df_test = pd.read_pickle('../data/Marie/Marie_test.pickle')
df_train = pd.read_pickle('../data/Marie/Marie_train.pickle')

X_test = df_test['utterance']
y_test = df_test['intent']

X = df_train['utterance']
y = df_train['intent']

model = nb.fit(X,y)
y_hat = model.predict(X_test)

prec_macro = precision_score(y_test, y_hat, average='macro')
rec_macro = recall_score(y_test,y_hat, average='macro')
f1_macro = f1_score(y_test,y_hat, average='macro')

results_full = pd.DataFrame(dict(
                    percent=[100],
                    test=[True],
                    recall=[rec_macro],
                    precision=[prec_macro],
                    F1=[f1_macro]))

print('recall macro: ' + str(rec_macro))
print('precision macro: ' + str(prec_macro))
print('F1 macro: ' + str(f1_macro))

# training score
y_hat_train = model.predict(X) 
prec_macro_train = precision_score(y, y_hat_train, average='macro')
rec_macro_train = recall_score(y,y_hat_train, average='macro')
f1_macro_train = f1_score(y,y_hat_train, average='macro')

print('recall macro train: ' + str(rec_macro_train))
print('precision macro train: ' + str(prec_macro_train))
print('F1 macro train: ' + str(f1_macro_train))

results_train = results_train.append(dict(
                    percent=100,
                    test=False,
                    recall=rec_macro_train,
                    precision=prec_macro_train,
                    F1=f1_macro_train), ignore_index=True)
##################################################################


### Varying training observations ########################################

for percent in [80,60,40,20]:
    for i in range(10):
        fn = '../data/Marie/train_obs/df_train_' + str(percent) +'_' + str(i)
        df = pd.read_pickle(fn)
        
        X = df['utterance']
        y = df['intent']
        
        model = nb.fit(X,y)
        y_hat = model.predict(X_test)

        prec_macro = precision_score(y_test, y_hat, average='macro')
        rec_macro = recall_score(y_test,y_hat, average='macro')
        f1_macro = f1_score(y_test,y_hat, average='macro')
        
        results_obs_test = results_obs.append(dict(
                    percent=percent,
                    test=True,
                    recall=rec_macro,
                    precision=prec_macro,
                    F1=f1_macro), ignore_index=True)

        # training score
        y_hat_train = model.predict(X) 
        prec_macro_train = precision_score(y, y_hat_train, average='macro')
        rec_macro_train = recall_score(y,y_hat_train, average='macro')
        f1_macro_train = f1_score(y,y_hat_train, average='macro')
        
        results_obs_train = results_train.append(dict(
                    percent=percent,
                    test=False,
                    recall=rec_macro_train,
                    precision=prec_macro_train,
                    F1=f1_macro_train), ignore_index=True)

results_train[results_train['test']==True].plot.scatter(x='percent',y='F1', ylim=(0,1))

results_train[results_train['test']==False].plot.scatter(x='percent',y='F1', ylim=(0,1))

results_train['model'] = 'Naive Bayes'

results_train[results_train['test']==True]

results_train.to_pickle('../results/results_train_nb.pickle')




### intents ##############################################

DIR = '../data/Marie/intent/'
results_intent = pd.DataFrame(columns=['intents','test','recall','precision','F1'])

for j in [5,10,15,20,25,30,35,40,45,50]:
    for i in range(10):
        
        fn_train = DIR +'df_train_' + str(j) + '_' + str(i)
        df_train = pd.read_pickle(fn_train)
        
#         fn_train_val = DIR +'df_train_val_' + str(j) + '_' + str(i)
#         df_train_val = pd.read_pickle(fn_train_val)
        
#         fn_val = DIR +'df_val_' + str(j) + '_' + str(i)
#         df_val = pd.read_pickle(fn_val)
        
        fn_test = DIR +'df_test_' + str(j) + '_' + str(i)
        df_test = pd.read_pickle(fn_test)
        
        X = df_train['utterance']
        y = df_train['intent']
        
        X_test = df_test['utterance']
        y_test = df_test['intent']
        
        model = nb.fit(X,y)
        y_hat = model.predict(X_test)

        prec_macro = precision_score(y_test, y_hat, average='macro')
        rec_macro = recall_score(y_test,y_hat, average='macro')
        f1_macro = f1_score(y_test,y_hat, average='macro')
        
        results_intent = results_intent.append(dict(
                    intents=j,
                    test=True,
                    recall=rec_macro,
                    precision=prec_macro,
                    F1=f1_macro), ignore_index=True)

        # training score
        y_hat_train = model.predict(X) 
        prec_macro_train = precision_score(y, y_hat_train, average='macro')
        rec_macro_train = recall_score(y,y_hat_train, average='macro')
        f1_macro_train = f1_score(y,y_hat_train, average='macro')
        
        results_intent = results_intent.append(dict(
                    intents = j,
                    test=False,
                    recall=rec_macro_train,
                    precision=prec_macro_train,
                    F1=f1_macro_train), ignore_index=True)

results_intent['intents'] = pd.to_numeric(results_intent['intents'])

results_intent[results_intent['test']==True].plot.scatter(x='intents',y='F1', ylim=(0,1))

results_intent[results_intent['test']==False].plot.scatter(x='intents',y='F1', ylim=(0,1))

results_intent['model'] = 'Naive Bayes'

results_intent.to_pickle('../results/results_intent_nb.pickle')


#########################################################
# Braun
########################################################

DIR = '../data/Braun/original/'

for data in ['travel', 'ubuntu', 'webapp']:
    fn_test = DIR + data + '_test.pickle'
    df_test = pd.read_pickle(fn_test)
    
    fn_train = DIR + data + '_train.pickle'
    df_train = pd.read_pickle(fn_train) # train or train2?

    X_test = df_test['utterance']
    y_test = df_test['intent']
    
    X = df_train['utterance']
    y = df_train['intent']

    model = nb.fit(X,y)
    y_hat = model.predict(X_test)

    prec_macro = precision_score(y_test, y_hat, average='macro')
    rec_macro = recall_score(y_test,y_hat, average='macro')
    f1_macro = f1_score(y_test,y_hat, average='macro')

    # training score
    y_hat_train = model.predict(X) 
    prec_macro_train = precision_score(y, y_hat_train, average='macro')
    rec_macro_train = recall_score(y,y_hat_train, average='macro')
    f1_macro_train = f1_score(y,y_hat_train, average='macro')
    
    results_train = pd.DataFrame(dict(
                    percent=[100],
                    test=[True],
                    recall=[rec_macro],
                    precision=[prec_macro],
                    F1=[f1_macro]))
    #     columns=['','recall','precision','F1'])

    print('recall macro: ' + str(rec_macro))
    print('precision macro: ' + str(prec_macro))
    print('F1 macro: ' + str(f1_macro))

    # pd.DataFrame(confusion_matrix(y_test, y_hat))

    # training score
    print('recall macro: ' + str(rec_macro))
    print('precision macro: ' + str(prec_macro))
    print('F1 macro: ' + str(f1_macro))

    results_train = results_train.append(dict(
                        percent=100,
                        test=False,
                        recall=rec_macro_train,
                        precision=prec_macro_train,
                        F1=f1_macro_train), ignore_index=True)
    
    results_train['model'] = 'Naive Bayes'
    
    results_train.to_pickle('../results/braun/results_'+ data +'_nb.pickle')

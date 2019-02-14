# experiments.py
#   called by main.py
#   parses config files
#   constructs datasets, models --> parses to training.py

# import torch

# import models.neural_cls
# from models.neural_cls.util import Loader, Trainer
# from models.neural_cls.models import BiLSTM
# from models.neural_cls.models import BiLSTM_BB

# import matplotlib.pyplot as plt

import pandas as pd
import json
from time import time
import os

from models.baseModel import BaseModel



class Experiment(object):
    def __init__(self, parameters):
        self.parameters = parameters
        self.meta = parameters
        self.base_model = BaseModel(parameters)
        # self.meta = {}
        # self.model = BaseModel(parameters)

    def run(self):
        if self.parameters['model'] == 'SVM':
            self.base_model.load_wordvectors()
        self.base_model.load_data(dataset_path= self.parameters['dataset_path'],
                                    wordpath= self.parameters['wordpath'],
                                    worddim = self.parameters['worddim'],
                                    train_fn= "train.pickle",
                                    test_fn= "test.pickle")
        self.base_model.load_model(wdim = self.parameters['worddim'],
                                    hdim=self.parameters['hdim'])
        self.F1_train, self.F1_test = self.base_model.train()

    def cv(self):
        if self.parameters['model'] in ['LSTM','BiLSTM','LSTM_BB','BiLSTM_BB']:
            count = 0
            k_fold = self.parameters['kfold']
            n_wdim = len(self.parameters['worddims'])
            n_hdim = len(self.parameters['hdims'])
            n_res = k_fold * n_wdim * n_hdim
            results_cv = pd.DataFrame(columns=['word_dim','h_dim','i','F1_train','F1_valid']) 
            # , index= range(0,n_res))
            cv_data_path = os.path.join(self.parameters['dataset_path'], 'cv')
            
            for w, worddim in enumerate(self.parameters['worddims']):
                # TODO bring loading of wordvectors before iter over k_fold
                #   only data is tokenization depends on i
                for i in range(k_fold):
                    
                    wordpath = self.parameters['wordpaths'][worddim]
                    # file names
                    train_fn = f"train_{i}.pickle"
                    valid_fn = f"valid_{i}.pickle"
        
                    res_path = self.parameters['cv_result_path']
                    
                    self.base_model.load_data(dataset_path= cv_data_path,
                                                wordpath= wordpath,
                                                worddim= worddim,
                                                train_fn= train_fn,
                                                test_fn= valid_fn)
                    
                    for h, hdim in enumerate(self.parameters['hdims']):
                        self.base_model.load_model(worddim, hdim)

                        F1_train, F1_valid = self.base_model.train()

                        # print(count)
                        
                        results_cv = results_cv.append(dict(word_dim=int(worddim),
                                        h_dim=int(hdim),
                                        i=i,
                                        F1_train=F1_train,
                                        F1_valid=F1_valid), ignore_index=True)
                        # results_cv.iloc[count] = dict(word_dim=worddim, h_dim=hdim,
                        #     i=i, F1_train=F1_train, F1_valid=F1_valid)
                        # print(results_cv.head())
                        # print(results_cv.tail())
                        count+=1

            # results_cv['word_dim'] = pd.to_numeric(results_cv['word_dim'], downcast='integer')
            # results_cv['h_dim'] = pd.to_numeric(results_cv['h_dim'], downcast='integer')
            # save raw results
            results_cv.to_pickle(os.path.join(self.parameters['cv_result_path'],'raw_results_cv.pickle'))

            df_res = results_cv.groupby(by=['word_dim','h_dim'])['F1_train','F1_valid'].mean()
            # print(df_res)
            df_res.to_csv(os.path.join(self.parameters['cv_result_path'],'df_res_cv.csv'))

            best = df_res[df_res['F1_valid']==df_res['F1_valid'].max()]
            self.best_w, self.best_h = best.index[0]

            print(f'Best word dim: {self.best_w}, Best hdim: {self.best_h}')

            return int(self.best_w), int(self.best_h)
        
            # Run model with best hyperparameters
            # self.base_model.load_data(dataset_path= self.parameters['dataset_path'],
            #                         wordpath= self.parameters['wordpath'],
            #                         worddim = self.best_w,
            #                         train_fn= f"{self.parameters['model']}_train.pickle",
            #                         test_fn= f"{self.parameters['model']}_test.pickle")
            # self.base_model.load_model(hdim=self.best_h)
            # self.losses, self.F1_train, self.F1_test = self.base_model.train()

    def intent(self):
        # TODO crossval in intent
        results_intent = pd.DataFrame(columns=['n_intent','i','F1_train','F1_test'])

        intent_path = os.path.join(self.parameters['dataset_path'], 'intent')
        for n_int in self.parameters['n_intents']:
            for i in range(self.parameters['int_runs']):
                train_fn = f'train_{n_int}_{i}.pickle'
                test_fn = f'test_{n_int}_{i}.pickle'

                # TODO 
                #   check results_path
                #   check data_path
                
                # This is the same as run
                if self.parameters['model'] == 'SVM':
                    self.base_model.load_wordvectors()
                self.base_model.load_data(dataset_path= intent_path,
                                    wordpath= self.parameters['wordpath'],
                                    worddim = self.parameters['worddim'],
                                    train_fn= train_fn,
                                    test_fn= test_fn)
                self.base_model.load_model(wdim = self.parameters['worddim'],
                                    hdim=self.parameters['hdim'])
                self.F1_train, self.F1_test = self.base_model.train()
                # End same as run

                results_intent = results_intent.append(dict(n_intent=n_int, i=i,
                    F1_train=self.F1_train,
                    F1_test=self.F1_test), ignore_index=True)

        results_intent.to_pickle(os.path.join(self.parameters['int_result_path'],
                                                'results_intent.pickle'))
        df_res = results_intent.groupby(by=['n_intent','i'])['F1_train','F1_test'].mean()
        df_res.to_csv(os.path.join(self.parameters['int_result_path'],'df_res_int.csv'))
        
        # TODO create a central folder to save F1 intents per model

    def vary_obs(self):
        N_OBS_LIST = [20,40,60,80]
        results_obs = pd.DataFrame(columns=['n_obs','i','F1_train','F1_test'])
        intent_path = os.path.join(self.parameters['dataset_path'], 'vary_obs')
        for n_obs in N_OBS_LIST:
            for i in range(self.parameters['obs_runs']):
                train_fn = f'train_{n_obs}_{i}.pickle'
                test_fn = f'test.pickle'

                # This is the same as run
                if self.parameters['model'] == 'SVM':
                    self.base_model.load_wordvectors()
                self.base_model.load_data(dataset_path= intent_path,
                                    wordpath= self.parameters['wordpath'],
                                    worddim = self.parameters['worddim'],
                                    train_fn= train_fn,
                                    test_fn= test_fn)
                self.base_model.load_model(wdim = self.parameters['worddim'],
                                    hdim=self.parameters['hdim'])
                self.F1_train, self.F1_test = self.base_model.train()
                # End same as run

                results_obs = results_obs.append(dict(n_obs=n_obs, i=i,
                    F1_train=self.F1_train,
                    F1_test=self.F1_test), ignore_index=True)

        results_obs.to_pickle(os.path.join(self.parameters['obs_result_path'],
                                                'results_obs.pickle'))
        df_res = results_obs.groupby(by=['n_obs','i'])['F1_train','F1_test'].mean()
        df_res.to_csv(os.path.join(self.parameters['obs_result_path'],'df_res_obs.csv'))

    def create_meta(self):
        # print(f'{self.losses}, {self.F1_train}, {self.F1_test}')
        # model_name = self.parameters['model']
        t = self.parameters['time']
        
        # meta['time'] = t
        # self.meta['best_w'] = self.best_w
        # self.meta['best_h'] = self.best_h

        # self.meta['losses'] = self.losses
        self.meta['F1_train'] = self.F1_train
        self.meta['F1_test'] = self.F1_test

        meta_path = os.path.join(self.parameters['result_path'], f'meta_{t}.json')
        with open(meta_path, 'w') as outfile:
            json.dump(self.meta, outfile)

        print(f'Done, experiment took: {round(time()-t, 2)}')
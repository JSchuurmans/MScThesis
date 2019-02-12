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

    def run(self, parameters):
        self.parameters = parameters
        self.base_model.load_data(dataset_path= self.parameters['dataset_path'],
                                    wordpath= self.parameters['wordpath'],
                                    worddim = self.parameters['worddim'],
                                    train_fn= f"{self.parameters['model']}_train.pickle",
                                    test_fn= f"{self.parameters['model']}_test.pickle")
        self.base_model.load_model(hdim=self.parameters['hdim'])
        self.losses, self.F1_train, self.F1_test = self.base_model.train()

    def cv(self):
        if self.parameters['model'] in ['LSTM','BiLSTM','LSTM_BB','BiLSTM_BB']:
            count = 0
            K_FOLD = 2
            n_wdim = len(self.parameters['worddims'])
            n_hdim = len(self.parameters['hdims'])
            n_res = K_FOLD * n_wdim * n_hdim
            results_cv = pd.DataFrame(columns=['word_dim','h_dim','i','F1_train','F1_valid'],
                                        index= range(0,n_res))

            for w, worddim in enumerate(self.parameters['worddims']):
                for i in range(K_FOLD):
                    
                    wordpath = self.parameters['wordpaths'][worddim]
                    # file names
                    train_fn = f"{self.parameters['dataset']}_train_{i}.pickle"
                    valid_fn = f"{self.parameters['dataset']}_valid_{i}.pickle"

                    cv_data_path = os.path.join(self.parameters['dataset_path'], 'cv')
                    
                    res_path = self.parameters['cv_result_path']

                    self.base_model.load_data(dataset_path= cv_data_path,
                                                wordpath= wordpath,
                                                worddim= worddim,
                                                train_fn= train_fn,
                                                test_fn= valid_fn)
                    
                    for h, hdim in enumerate(self.parameters['hdims']):
                        self.base_model.load_model(hdim)

                        losses, F1_train, F1_test = self.base_model.train()

                        results_cv.iloc[count] = dict(word_dim=worddim, h_dim=h_dim,
                            i=i, F1_train=F1_train, F1_valid=F1_valid)

                        count+=1

            results_cv['word_dim'] = pd.to_numeric(results_cv['word_dim'], downcast='integer')
            results_cv['h_dim'] = pd.to_numeric(results_cv['h_dim'], downcast='integer')
            # save raw results
            results_cv.to_pickle(os.path.join(self.parameters['cv_result_path'],'raw_results_cv.pickle'))

            df_res = results_cv.groupby(by=['word_dim','h_dim'])['F1_train','F1_valid'].mean()
            # print(df_res)
            df_res.to_csv(os.path.join(self.parameters['cv_result_path'],'df_res_cv.csv'))

            best = df_res[df_res['F1_valid']==df_res['F1_valid'].max()]
            self.best_w, self.best_h = best.index[0]

            return self.best_w, self.best_h
        
            # Run model with best hyperparameters
            # self.base_model.load_data(dataset_path= self.parameters['dataset_path'],
            #                         wordpath= self.parameters['wordpath'],
            #                         worddim = self.best_w,
            #                         train_fn= f"{self.parameters['model']}_train.pickle",
            #                         test_fn= f"{self.parameters['model']}_test.pickle")
            # self.base_model.load_model(hdim=self.best_h)
            # self.losses, self.F1_train, self.F1_test = self.base_model.train()

            

    def create_meta(self):
        print(f'{self.losses}, {self.F1_train}, {self.F1_test}')
        # model_name = self.parameters['model']
        t = parameters['time']
        
        # meta['time'] = t
        self.meta['best_w'] = self.best_w
        self.meta['best_h'] = self.best_h

        meta['losses'] = self.losses
        meta['F1_train'] = self.F1_train
        meta['F1_test'] = self.F1_test

        meta_path = os.path.join(self.parameters['result_path'], f'meta_{t}.json')
        with open(meta_path, 'w') as outfile:
            json.dump(meta, outfile)

        print(f'Done, experiment took: {round(time()-t, 2)}')
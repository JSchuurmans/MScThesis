# model.py
#   called by main.py
#   parses config files
#   constructs datasets, models --> parses to training.py
import torch

import os
from time import time
import json
import pandas as pd
import matplotlib.pyplot as plt

# neural modules
import models.neural_cls
from models.neural_cls.util import Loader, Trainer
from models.neural_cls.models import BiLSTM
from models.neural_cls.models import BiLSTM_BB

# naive bayes module
from models.naive_bayes import nb_pipe

# svm
from datasets.utils import *
from models.svm import * # svm_grid, experiment_train, experiment_intent
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

class BaseModel(object):
    def __init__(self, parameters):
        self.parameters = parameters
        # self.meta = {}
        # self.model = BaseModel(parameters)
        self.word_vectors = None

    def load_wordvectors(self):
        # TODO create propper paths
        # TODO create re-locate wordvector files to wordvector directory
        if self.parameters['word_vector'] == 'ft':
            self.word_vectors = KeyedVectors.load_word2vec_format('C:\Local_data\wiki.en\wiki.en.vec', binary=False)
        elif self.parameters['word_vector'] == 'gl':
            # if gensim format of GloVe does not exists
            if not os.path.exists('C:\Local_data\glove.6B\gensim_glove.6B.300d.txt'):
                # convert GloVe to gensim format
                glove2word2vec(glove_input_file='C:\Local_data\glove.6B\glove.6B.300d.txt', word2vec_output_file='C:\Local_data\glove.6B\gensim_glove.6B.300d.txt')
            self.word_vectors = KeyedVectors.load_word2vec_format('C:\Local_data\glove.6B\gensim_glove.6B.300d.txt', binary=False)
        elif self.parameters['word_vector'] == 'wv':
            self.word_vectors = KeyedVectors.load_word2vec_format('C:\Local_data\GoogleNews-vectors\GoogleNews-vectors-negative300.bin', binary=True)
        else:
            raise NotImplementedError

    def load_data(self, dataset_path, wordpath=None, worddim=None, 
                    train_fn='braun_train.pickle', test_fn='braun_test.pickle', label='intent'):
        
        if self.parameters['model'] in ['LSTM','BiLSTM', 'LSTM_BB','BiLSTM_BB']:
            loader = Loader()

            if self.parameters['dataset'] == 'mareview':
                self.train_data, self.test_data, mappings = loader.load_mareview(
                                                            dataset_path, 
                                                            wordpath, 
                                                            worddim)
            elif self.parameters['dataset'] in ['braun','retail','travel','ubuntu','webapp2']:
                self.train_data, self.test_data, mappings = loader.load_pickle(
                                                            dataset_path, 
                                                            wordpath, 
                                                            worddim,
                                                            train_fn,
                                                            test_fn,
                                                            label,
                                                            self.word_vectors)

            self.word_to_id = mappings['word_to_id']
            self.tag_to_id = mappings['tag_to_id']
            self.word_embeds = mappings['word_embeds']
        
        elif self.parameters['model'] in ['NB','SVM']:
            train_path = os.path.join(dataset_path, train_fn)
            test_path = os.path.join(dataset_path, test_fn)

            df_train = pd.read_pickle(train_path)
            df_test = pd.read_pickle(test_path)

            self.X_train = df_train['utterance'] # TODO make dynamic/note in README
            self.y_train = df_train[label]

            self.X_test = df_test['utterance'] 
            self.y_test = df_test[label]

            # data loading for SVM
            if self.parameters['model'] =='SVM':
                texts_train = remove_stopwords(remove_non_alpha_num(self.X_train))
                texts_test = remove_stopwords(remove_non_alpha_num(self.X_test))

                self.train_sum, self.train_ave = create_input(texts_train, self.word_vectors)
                self.test_sum, self.test_ave = create_input(texts_test, self.word_vectors)

        print('Loading Dataset Complete')

    def load_model(self, wdim, hdim):
        if self.parameters['model'] in ['LSTM','BiLSTM','LSTM_BB','BiLSTM_BB']:
            if self.parameters['reload']:
                print ('Loading Saved Weights....................................................................')
                model_path = os.path.join(self.parameters['checkpoint_path'], 'modelweights')
                self.model = torch.load(model_path)
            else:
                print('Building Model............................................................................')
                word_vocab_size = len(self.word_to_id)
                # word_embedding_dim = self.parameters['worddim']
                bidirectional = self.parameters['bidir']
                output_size = len(self.tag_to_id)
                # Build NN
                if self.parameters['model'][-4:] == 'LSTM':
                    print (f"(Bi: {self.parameters['bidir']}) LSTM")
                    self.model = BiLSTM(word_vocab_size, wdim, 
                            hdim, output_size, 
                            pretrained = self.word_embeds,
                            bidirectional = bidirectional)
                # Build Bayesian NN
                elif self.parameters['model'][-2:] == 'BB':
                    print (f"(Bi: {self.parameters['bidir']}) LSTM_BB")
                    sigma_prior = self.parameters['sigmp']
                    self.model = BiLSTM_BB(word_vocab_size, wdim, 
                            hdim, output_size, 
                            pretrained = self.word_embeds,
                            bidirectional = bidirectional, 
                            sigma_prior=sigma_prior)
            self.model.cuda()
        
        elif self.parameters['model'] == 'NB':
            self.model = nb_pipe(self.parameters['ngram'])
        
        elif self.parameters['model'] == 'SVM':
            self.model = svm_grid

    def train(self):
        meta_data = {}
        if self.parameters['model'] in ['LSTM','BiLSTM','LSTM_BB','BiLSTM_BB']:
            learning_rate = self.parameters['lrate']
            num_epochs = self.parameters['nepch']
            print(f'Initial learning rate is: {learning_rate}')
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            # load trainer
            # if self.parameters['model'][-4:] == 'LSTM':
            trainer = Trainer(self.model, optimizer, self.parameters['result_path'], self.parameters['model'], 
                                self.tag_to_id, usedataset= self.parameters['dataset'])
            # load Bayesian trainer
            # elif self.parameters['model'][-2:] == 'BB':
                # trainer = Trainer3(self.model, optimizer, self.parameters['result_path'], self.parameters['model'], 
                                    # self.tag_to_id, usedataset= self.parameters['dataset'])
            # train (Bayesian) NN
            self.losses, _, all_P, all_R, self.all_F = trainer.train_model(num_epochs, self.train_data, 
                                    self.test_data, learning_rate,
                                    batch_size = self.parameters['batch_size'],
                                    checkpoint_path = self.parameters['checkpoint_path'])
            # TODO self.losses write losses away to log file
            F1_train = self.all_F[-1][0]
            F1_test = self.all_F[-1][1]

            P_train = all_P[-1][0]
            P_test = all_P[-1][1]
            
            R_train = all_R[-1][0]
            R_test = all_R[-1][1]

            meta_data['losses'] = self.losses
            # meta_data['all_F'] = self.all_F
            # meta_data['all_P'] = all_P
            # meta_data['all_R'] = all_R

            # return F1_train, F1_test

        elif self.parameters['model'] == 'NB':
            self.fitted_model = self.model.fit(self.X_train, self.y_train)
            
            y_hat = self.fitted_model.predict(self.X_test)
            P_test = precision_score(self.y_test, y_hat, average='macro')
            R_test = recall_score(self.y_test, y_hat, average='macro')
            F1_test = f1_score(self.y_test, y_hat, average='macro')

            print('recall macro test: ' + str(R_test))
            print('precision macro test: ' + str(P_test))
            print('F1 macro test: ' + str(F1_test))

            y_hat_train = self.fitted_model.predict(self.X_train) 
            P_train = precision_score(self.y_train, y_hat_train, average='macro')
            R_train = recall_score(self.y_train, y_hat_train, average='macro')
            F1_train = f1_score(self.y_train, y_hat_train, average='macro')

            print('recall macro train: ' + str(R_train))
            print('precision macro train: ' + str(P_train))
            print('F1 macro train: ' + str(F1_train))

        elif self.parameters['model'] == 'SVM':
            cbow = self.parameters['cbow']
            if cbow == 'sum':
                R,P, F1, cv_res, _ = self.model(self.train_sum, self.y_train, self.test_sum, self.y_test, self.parameters['kfold'])
            elif cbow == 'ave':
                R,P, F1, cv_res, _ = self.model(self.train_ave, self.y_train, self.test_ave, self.y_test, self.parameters['kfold'])
            
            F1_train = F1[1]
            F1_test = F1[0]
            P_train = P[1]
            P_test = P[0]
            R_train = R[1]
            R_test = R[0]

            meta_data['best_param'] = cv_res['best_param']
            meta_data['best_score'] = cv_res['best_score']
        
        train_results = {'F1':F1_train, 'P':P_train, 'R':R_train}
        test_results = {'F1':F1_test, 'P':P_test, 'R':R_test}
        return meta_data, train_results, test_results

    def test(self):
        raise NotImplementedError

    def plot_loss(self):
        plt.plot(self.losses)
        plt.savefig(os.path.join(self.parameters['result_path'], 'lossplot.png'))

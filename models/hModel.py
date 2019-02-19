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
from models.neural_cls.models import h_BiLSTM

from datasets.retail.retail_data_hierarchy import retail_data_hierarchy
from datasets.braun.braun_data_hierarchy import braun_data_hierarchy

# naive bayes module
from models.naive_bayes import stemming_tokenizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import string

# svm
from datasets.utils import *
from models.svm import * # svm_grid, experiment_train, experiment_intent
from models.svm_best import svm_best
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

class HModel(object):
    def __init__(self, parameters):
        self.parameters = parameters
        # self.meta = {}
        # self.model = BaseModel(parameters)
        # super(BaseModel)

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
                    train_fn='train.pickle', test_fn='test.pickle', label='intent'):
        
        # load original data
        ori_train_path = os.path.join(dataset_path, train_fn)
        ori_test_path = os.path.join(dataset_path, test_fn)

        df_train = pd.read_pickle(ori_train_path)
        df_test = pd.read_pickle(ori_test_path)

        if self.parameters['dataset'] == 'braun':
            self.n_nodes, h_path = braun_data_hierarchy(dataset_path, train_fn, test_fn)
        elif self.parameters['dataset'] == 'retail':
            self.n_nodes, h_path = retail_data_hierarchy(dataset_path, train_fn, test_fn)
        
        dataset_path = h_path
        
        if self.parameters['model'] in ['LSTM','BiLSTM', 'LSTM_BB','BiLSTM_BB']:
            label = 'intent'
            loader = Loader()

            if self.parameters['dataset'] in ['braun','retail','travel','ubuntu','webapp2']:
                self.train_data, self.test_data, mappings = loader.load_pickle(
                                                            dataset_path, 
                                                            wordpath, 
                                                            worddim,
                                                            train_fn,
                                                            test_fn,
                                                            label)

            self.word_to_id = mappings['word_to_id']
            self.tag_to_id = mappings['tag_to_id']
            print(len(self.tag_to_id))
            print(self.tag_to_id)
            self.word_embed = mappings['word_embeds']
        
        elif self.parameters['model'] in ['NB','SVM']:
            label = 'category'

            train_path = os.path.join(dataset_path, train_fn)
            test_path = os.path.join(dataset_path, test_fn)

            df_train = pd.read_pickle(train_path)
            df_test = pd.read_pickle(test_path)

            self.X_train = df_train['utterance'] # TODO make dynamic/note in README
            self.y_train = df_train[label]
            self.true_train = df_train['intent']

            self.X_test = df_test['utterance'] 
            self.y_test = df_test['intent'] # label

            # data loading for SVM
            if self.parameters['model'] =='SVM':
                texts_train = remove_stopwords(remove_non_alpha_num(self.X_train))
                texts_test = remove_stopwords(remove_non_alpha_num(self.X_test))

                train_sum, train_ave = create_input(texts_train, self.word_vectors)
                test_sum, test_ave = create_input(texts_test, self.word_vectors)
                if self.parameters['cbow'] =='sum':
                    self.X_train, self.X_test = train_sum, test_sum
                elif self.parameters['cbow'] == 'ave':
                    self.X_train, self.X_test = train_ave, test_ave

        ##### hierarchical ###############################################################################################
        label = 'intent' # subclass
        self.train_datas = []
        self.test_datas = []
        self.word_to_ids = []
        self.tag_to_ids = []
        self.word_embeds = []
        self.X_trains = {}
        self.y_trains = {}
        self.X_tests = {}
        self.y_tests = {}
        train_sums = {}
        train_aves = {}
        test_sums = {}
        test_aves = {}
        for i in range(self.n_nodes):
            # print(i)
            h_train_fn = f'{i}_{train_fn}'
            h_test_fn = f'{i}_{test_fn}'
            # if self.parameters['model'] in ['LSTM','BiLSTM', 'LSTM_BB','BiLSTM_BB']:
            #     # loader = Loader()
                
            #     if self.parameters['dataset'] in ['braun','retail','travel','ubuntu','webapp2']:
            #         self.train_data[i], self.test_data[i], mappings = loader.load_pickle(
            #                                                     dataset_path, 
            #                                                     wordpath, 
            #                                                     worddim,
            #                                                     h_train_fn,
            #                                                     h_test_fn,
            #                                                     label)
            #     self.word_to_ids[i] = mappings['word_to_id']
            #     self.tag_to_ids[i] = mappings['tag_to_id']
            #     self.word_embeds[i] = mappings['word_embeds']
            
            if self.parameters['model'] in ['NB','SVM']:
                train_path = os.path.join(dataset_path, h_train_fn)
                test_path = os.path.join(dataset_path, h_test_fn)

                df_train = pd.read_pickle(train_path)
                df_test = pd.read_pickle(test_path)

                self.X_trains[i] = df_train['utterance'] # TODO make dynamic/note in README
                self.y_trains[i] = df_train[label]

                self.X_tests[i] = df_test['utterance'] 
                self.y_tests[i] = df_test[label]

                # data loading for SVM
                if self.parameters['model'] =='SVM':
                    texts_train = remove_stopwords(remove_non_alpha_num(self.X_trains[i]))
                    texts_test = remove_stopwords(remove_non_alpha_num(self.X_tests[i]))

                    train_sums[i], train_aves[i] = create_input(texts_train, self.word_vectors)
                    test_sums[i], test_aves[i] = create_input(texts_test, self.word_vectors)
                    if self.parameters['cbow'] =='sum':
                        self.X_trains[i], self.X_tests[i] = train_sums[i], test_sums[i]
                    elif self.parameters['cbow'] == 'ave':
                        self.X_trains[i], self.X_tests[i] = train_aves[i], test_aves[i]
       
        ################################################################################################
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
                output_size = len(self.tag_to_id)
                bidirectional = self.parameters['bidir']
                # Build NN
                if self.parameters['model'][-4:] == 'LSTM':
                    print (f"(Bi: {self.parameters['bidir']}) LSTM")
                    self.model = h_BiLSTM(word_vocab_size, wdim, 
                            hdim, output_size, 
                            pretrained = self.word_embed,
                            bidirectional = bidirectional,
                            tag_to_id = self.tag_to_id,
                            dataset = self.parameters['dataset']) #TODO hierarchy
                # Build Bayesian NN
                elif self.parameters['model'][-2:] == 'BB':
                    print (f"(Bi: {self.parameters['bidir']}) LSTM_BB")
                    sigma_prior = self.parameters['sigmp']
                    self.model = h_BiLSTM_BB(word_vocab_size, wdim, 
                            hdim, output_size, 
                            pretrained = self.word_embed,
                            bidirectional = bidirectional, 
                            sigma_prior=sigma_prior)
            self.model.cuda()
        
        elif self.parameters['model'] == 'NB':
            self.model = Pipeline([
                            ('vectorizer', TfidfVectorizer(tokenizer=stemming_tokenizer,
                                   stop_words=stopwords.words('english')+ list(string.punctuation),
                                   min_df=3)),
                            ('classifier', MultinomialNB(alpha=1)),])
        
        elif self.parameters['model'] == 'SVM':
            self.model,_ = svm_best(self.X_train,self.y_train)

        ### hierarchical ######################################################################
        self.models = {}
        for i in range(self.n_nodes):
            # if self.parameters['model'] in ['LSTM','BiLSTM','LSTM_BB','BiLSTM_BB']:
            #     raise NotImplementedError
            if self.parameters['model'] == 'NB':
                self.models[i] = Pipeline([
                            ('vectorizer', TfidfVectorizer(tokenizer=stemming_tokenizer,
                                   stop_words=stopwords.words('english')+ list(string.punctuation),
                                   min_df=3)),
                            ('classifier', MultinomialNB(alpha=1)),])
                # print(self.models[i])
            elif self.parameters['model'] == 'SVM':
                self.models[i], _ = svm_best(self.X_trains[i], self.y_trains[i])
                
                # svm.SVC(class_weight='balanced', kernel='rbf', gamma='1000', C=1) # TODO parameters

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
            self.losses, _, all_P, all_R, self.all_F = trainer.train_model(num_epochs, self.train_data, self.test_data, learning_rate,
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

        elif self.parameters['model'] in ['NB','SVM']:
            # print(self.y_train.unique())
            self.fitted_model = self.model.fit(self.X_train, self.y_train)

            ### hierarchical
            self.fitted_models = {}
            for i in range(self.n_nodes):
                self.fitted_models[i] = self.models[i].fit(self.X_trains[i], self.y_trains[i])


            ## prediction #############
            # print(self.X_test)
            cat_hat = self.fitted_model.predict(self.X_test)
            # print(cat_hat)

            # y_hat = [None] * len(cat_hat)

            # print(self.X_test[1])

            y_hat = [self.fitted_models[row].predict([self.X_test[r]]) for r,row in enumerate(cat_hat)]

            # for r, row in enumerate(cat_hat):
            #     print(self.X_test[r])
            #     y_hat[r] = self.fitted_models[row].predict([self.X_test[r]])

            P_test = precision_score(self.y_test, y_hat, average='macro')
            R_test = recall_score(self.y_test, y_hat, average='macro')
            F1_test = f1_score(self.y_test, y_hat, average='macro')

            print('recall macro test: ' + str(R_test))
            print('precision macro test: ' + str(P_test))
            print('F1 macro test: ' + str(F1_test))
            

            # train score
            cat_hat_train = self.fitted_model.predict(self.X_train)
            
            y_hat_train = [self.fitted_models[row].predict([self.X_train[r]]) for r,row in enumerate(cat_hat_train)]
            
            P_train = precision_score(self.true_train, y_hat_train, average='macro')
            R_train = recall_score(self.true_train, y_hat_train, average='macro')
            F1_train = f1_score(self.true_train, y_hat_train, average='macro')

            print('recall macro train: ' + str(R_train))
            print('precision macro train: ' + str(P_train))
            print('F1 macro train: ' + str(F1_train))
        
        train_results = {'F1':F1_train, 'P':P_train, 'R':R_train}
        test_results = {'F1':F1_test, 'P':P_test, 'R':R_test}
        return meta_data, train_results, test_results

    def test(self):
        raise NotImplementedError

    def plot_loss(self):
        plt.plot(self.losses)
        plt.savefig(os.path.join(self.parameters['result_path'], 'lossplot.png'))

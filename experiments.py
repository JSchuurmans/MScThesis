# experiments.py
#   called by main.py
#   parses config files
#   constructs datasets, models --> parses to training.py
import torch

import os

import models.neural_cls
from models.neural_cls.util import Loader, Trainer
from models.neural_cls.models import BiLSTM
from models.neural_cls.models import BiLSTM_BB
import matplotlib.pyplot as plt

from time import time
import json

class Experiment(object):
    def __init__(self, parameters):
        self.parameters = parameters
        # self.meta = {}
        # self.model = BaseModel(parameters)

    def load_data(self):
        
        if self.parameters['model'] in ['LSTM','BiLSTM', 'LSTM_BB','BiLSTM_BB']:
            loader = Loader()

            if self.parameters['dataset'] == 'mareview':
                self.train_data, self.test_data, mappings = loader.load_mareview(
                                                            self.parameters['dataset_path'], 
                                                            self.parameters['pretrnd'], 
                                                            self.parameters['worddim'])
            elif self.parameters['dataset'] in ['braun','retail']:
                self.train_data, self.test_data, mappings = loader.load_pickle(
                                                            self.parameters['dataset_path'], 
                                                            self.parameters['pretrnd'], 
                                                            self.parameters['worddim'])

            self.word_to_id = mappings['word_to_id']
            self.tag_to_id = mappings['tag_to_id']
            self.word_embeds = mappings['word_embeds']
        
        else:
            raise NotImplementedError()

        print('Loading Dataset Complete')

    def load_model(self):
        if self.parameters['model'] in ['LSTM','BiLSTM','LSTM_BB','BiLSTM_BB']:
            if self.parameters['reload']:
                print ('Loading Saved Weights....................................................................')
                model_path = os.path.join(self.parameters['checkpoint_path'], 'modelweights')
                self.model = torch.load(model_path)
            else:
                print('Building Model............................................................................')
                word_vocab_size = len(self.word_to_id)
                word_embedding_dim = self.parameters['worddim']
                word_hidden_dim = self.parameters['hdim']
                output_size = self.parameters['opsiz']
                bidirectional = self.parameters['bidir']
                # Build NN
                if self.parameters['model'][-4:] == 'LSTM':
                    print (f"(Bi: {self.parameters['bidir']}) LSTM")
                    self.model = BiLSTM(word_vocab_size, word_embedding_dim, 
                            word_hidden_dim, output_size, 
                            pretrained = self.word_embeds,
                            bidirectional = bidirectional)
                # Build Bayesian NN
                elif self.parameters['model'][-2:] == 'BB':
                    print (f"(Bi: {self.parameters['bidir']}) LSTM_BB")
                    sigma_prior = self.parameters['sigmp']
                    self.model = BiLSTM_BB(word_vocab_size, word_embedding_dim, 
                            word_hidden_dim, output_size, 
                            pretrained = self.word_embeds,
                            bidirectional = bidirectional, 
                            sigma_prior=sigma_prior)
            self.model.cuda()

    def train(self):
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
            self.losses,_,_,_, self.all_F = trainer.train_model(num_epochs, self.train_data, self.test_data, learning_rate,
                                    batch_size = self.parameters['batch_size'],
                                    checkpoint_path = self.parameters['checkpoint_path'])

    def test(self):
        raise NotImplementedError
    
    def create_meta(self):
        meta = self.parameters
        # model_name = self.parameters['model']
        t = time()
        meta['time'] = t
        meta_path = os.path.join(self.parameters['result_path'], f'meta_{t}.json')
        with open(meta_path, 'w') as outfile:
            json.dump(meta, outfile)

    def plot_loss(self):
        plt.plot(self.losses)
        plt.savefig(os.path.join(self.parameters['result_path'], 'lossplot.png'))
    

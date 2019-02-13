# main.py
#   command line interface
#   creates experiment object from experiments.py
#   change hyperparameters (incl dataset and model) via cmd line or config files

from __future__ import print_function
# from collections import OrderedDict
import os
import numpy as np
from time import time

import argparse

from experiments import Experiment

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', action='store', dest='dataset', default='braun', type=str,
                    help='Dataset to be Used')
parser.add_argument('--log_path', action='store', dest='log_path', default='log',
                    type=str, help='Path to Save/Load Result')
parser.add_argument('--model', action='store', dest='model', default='BiLSTM', type=str, 
                    help='Model to Use')
parser.add_argument('--worddim', default=300, type=int, dest='worddim',
                    help="Word Embedding Dimension")
parser.add_argument('--wordpath', default="wordvectors/glove.6B.300d.txt", dest='wordpath',
                    type=str, help="Location of pretrained embeddings")
parser.add_argument('--reload', default=0, type=int, dest='reload',
                    help="Reload the last saved model")
parser.add_argument('--checkpoint', default="checkpoint", type=str, dest='checkpoint',
                    help="Location of trained Model")
parser.add_argument('--crossval', action='store',  dest='crossval', default=0, 
                    type=int, help="Location of trained Model")

opt = parser.parse_args()

parameters = vars(opt)

use_dataset = parameters['dataset']
model_name = parameters['model']
parameters['time'] = time()

WORD_PATHS = {50:"wordvectors/glove.6B.50d.txt",
        100:"wordvectors/glove.6B.100d.txt",
        200:"wordvectors/glove.6B.200d.txt",
        300:"wordvectors/glove.6B.300d.txt"}


if parameters['crossval']:
    parameters['wordpaths'] = WORD_PATHS
    parameters['worddims'] = WORD_PATHS.keys()
    parameters['hdims'] = [25,50,75,100,150,200,300]
    parameters['cv_result_path'] = os.path.join(opt.log_path, use_dataset,'cv', f"{model_name}",f"{parameters['time']}")
    if not os.path.exists(parameters['cv_result_path']):
        os.makedirs(parameters['cv_result_path'])

parameters['dataset_path'] = os.path.join('datasets', use_dataset)
parameters['checkpoint_path'] = os.path.join(opt.log_path, use_dataset, model_name, parameters['checkpoint'])
parameters['result_path'] = os.path.join(opt.log_path, use_dataset, f"{model_name}_{parameters['time']}")

# loader = Loader()

print('Model:', model_name)
print('Dataset:', use_dataset)

if not os.path.exists(parameters['result_path']):
    os.makedirs(parameters['result_path'])

if not os.path.exists(parameters['checkpoint_path']):
    os.makedirs(parameters['checkpoint_path'])  



# if not os.path.exists(os.path.join(parameters['checkpoint_path'], 'modelweights')):
#     os.makedirs(os.path.join(parameters['checkpoint_path'], 'modelweights'))

# model specific parameters
if opt.model == 'LSTM':

    parameters['bidir'] = False

    if opt.dataset == 'braun':
        parameters['dpout'] = 0.5
        parameters['hdim'] = 100
        parameters['nepch'] = 25 # 10 for trec
        
        parameters['lrate'] = 0.001
        parameters['batch_size'] = 10
        parameters['opsiz'] = 14 # 6 for trec2

    elif opt.dataset == 'retail':
        parameters['dpout'] = 0.5
        parameters['hdim'] = 150
        parameters['nepch'] = 25 # 10 for trec
        
        parameters['lrate'] = 0.001
        parameters['batch_size'] = 10
        parameters['opsiz'] = 54 # 6 for trec2

elif opt.model == 'BiLSTM':
    parameters['bidir'] = True

    if opt.dataset == 'mareview':
        parameters['dpout'] = 0.5
        parameters['hdim'] = 200
        parameters['nepch'] = 5 # 10 for trec
        
        parameters['lrate'] = 0.001
        parameters['batch_size'] = 50
        parameters['opsiz'] = 2 # 6 for trec2
    
    elif opt.dataset == 'braun':
        parameters['dpout'] = 0.5
        parameters['hdim'] = 150
        parameters['nepch'] = 25 # 10 for trec
        
        parameters['lrate'] = 0.001
        parameters['batch_size'] = 10
        parameters['opsiz'] = 14 # 6 for trec2

    elif opt.dataset == 'retail':
        parameters['dpout'] = 0.5
        parameters['hdim'] = 150
        parameters['nepch'] = 25 # 10 for trec
        
        parameters['lrate'] = 0.001
        parameters['batch_size'] = 10
        parameters['opsiz'] = 54 # 6 for trec2

elif opt.model == 'LSTM_BB':

    parameters['bidir'] = False
    parameters['sigmp'] = float(np.exp(-3))

    if opt.dataset == 'braun':
        parameters['dpout'] = 0.5
        parameters['hdim'] = 100
        parameters['nepch'] = 25 # 10 for trec
        
        parameters['lrate'] = 0.001
        parameters['batch_size'] = 10
        parameters['opsiz'] = 14 # 6 for trec2

    elif opt.dataset == 'retail':
        parameters['dpout'] = 0.5
        parameters['hdim'] = 150
        parameters['nepch'] = 25 # 10 for trec
        
        parameters['lrate'] = 0.001
        parameters['batch_size'] = 10
        parameters['opsiz'] = 54 # 6 for trec2

elif opt.model == 'BiLSTM_BB':
    parameters['bidir'] = True
    parameters['sigmp'] = float(np.exp(-3))

    if opt.dataset == 'mareview':
        parameters['dpout'] = 0.5
        parameters['hdim'] = 200
        parameters['nepch'] = 5 # 10 for trec
        
        parameters['lrate'] = 0.001
        parameters['batch_size'] = 50
        parameters['opsiz'] = 2 # 6 for trec2
    
    elif opt.dataset == 'braun':
        parameters['dpout'] = 0.5
        parameters['hdim'] = 100
        parameters['nepch'] = 25 # 10 for trec
        
        parameters['lrate'] = 0.001
        parameters['batch_size'] = 10
        parameters['opsiz'] = 14 # 6 for trec2

    elif opt.dataset == 'retail':
        parameters['dpout'] = 0.5
        parameters['hdim'] = 150
        parameters['nepch'] = 25 # 10 for trec
        
        parameters['lrate'] = 0.001
        parameters['batch_size'] = 10
        parameters['opsiz'] = 54 # 6 for trec2


elif opt.model == 'NB' and opt.dataset == 'braun':
    print(f'Model: {opt.model}, Dataset: {opt.dataset}')

elif opt.model == 'NB' and opt.dataset == 'retail':
    print(f'Model: {opt.model}, Dataset: {opt.dataset}')

elif opt.model == 'SVM' and opt.dataset == 'braun':
    print(f'Model: {opt.model}, Dataset: {opt.dataset}')

elif opt.model == 'SVM' and opt.dataset == 'retail':
    print(f'Model: {opt.model}, Dataset: {opt.dataset}')
    # parameters['sigmp'] = float(np.exp(-3))
else:
    raise NotImplementedError()




experiment = Experiment(parameters)

if parameters['crossval']:
    # parameters = 
    best_w, best_h = experiment.cv()
    parameters['worddim'] = best_w
    parameters['wordpath'] = WORD_PATHS[best_w]
    parameters['hdim'] = best_h

# parse parameters again,
#   so they get saved in meta
#   and worddim and hdim get overwritten with best
experiment.run(parameters)

experiment.create_meta()

# if opt.model == 'BiLSTM':
#     experiment.plot_loss()
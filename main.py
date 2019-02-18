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
parser.add_argument('--hdim', default=150, type=int, dest='hdim',
                    help="Hidden Dimension")
parser.add_argument('--wordpath', default="wordvectors/glove.6B.300d.txt", dest='wordpath',
                    type=str, help="Location of pretrained embeddings")
parser.add_argument('--reload', default=0, type=int, dest='reload',
                    help="Reload the last saved model")
parser.add_argument('--checkpoint', default="checkpoint", type=str, dest='checkpoint',
                    help="Location of trained Model")
parser.add_argument('--full', action='store',  dest='full', default=1, 
                    type=int, help="Location of trained Model")
parser.add_argument('--crossval', action='store',  dest='crossval', default=0, 
                    type=int, help="Cross validate LSTM")
parser.add_argument('--intent', action='store',  dest='intent', default=0, 
                    type=int, help="Vary number of intents")
parser.add_argument('--vary_obs', action='store',  dest='vary_obs', default=0, 
                    type=int, help="Vary number of training observations")
parser.add_argument('--word_vector', action='store',  dest='word_vector', default='ft', 
                    type=str, help="Type of wordvectors")
parser.add_argument('--cbow', action='store',  dest='cbow', default='sum', 
                    type=str, help="Type of wordvectors")
parser.add_argument('--hier', action='store',  dest='hier', default=0, 
                    type=int, help="User Hierarchical Model")


opt = parser.parse_args()

parameters = vars(opt)

use_dataset = parameters['dataset']
# model_name = parameters['model']
if parameters['model'] == 'SVM':
    model_name = f"{parameters['model']}_{parameters['word_vector']}_{parameters['cbow']}"
else:
    model_name = parameters['model']
if parameters['hier']:
    model_name = f'h_{model_name}'
parameters['model_name'] = model_name

parameters['time'] = time()

WORD_PATHS = {50:"wordvectors/glove.6B.50d.txt",
        100:"wordvectors/glove.6B.100d.txt",
        200:"wordvectors/glove.6B.200d.txt",
        300:"wordvectors/glove.6B.300d.txt"}

INTENT_RUNS = 10
OBS_RUNS = 10

if parameters['crossval']:
    if parameters['dataset'] in ['braun','travel','ubuntu','webapp','webapp2']:
        parameters['kfold'] = 2
    elif parameters['dataset'] == 'retail':
        parameters['kfold'] = 5
    parameters['wordpaths'] = WORD_PATHS
    parameters['worddims'] = WORD_PATHS.keys()
    parameters['hdims'] = [25,50,75,100,150,200,300]
    parameters['cv_result_path'] = os.path.join(opt.log_path, use_dataset,'cv', model_name, str(parameters['time']))
    if not os.path.exists(parameters['cv_result_path']):
        os.makedirs(parameters['cv_result_path'])

if parameters['intent']:
    parameters['int_runs'] = INTENT_RUNS
    parameters['int_result_path'] = os.path.join(opt.log_path, use_dataset,'intent', model_name, str(parameters['time']))
    if not os.path.exists(parameters['int_result_path']):
        os.makedirs(parameters['int_result_path'])
    if parameters['dataset'] == 'braun':
        parameters['n_intents'] =  range(5,14) # TODO range(2,14) gives error for crossval SVM
    if parameters['dataset'] == 'retail':
        parameters['n_intents'] = [5,10,15,20,25,30,35,40,45,50]

if parameters['vary_obs']:
    parameters['obs_runs'] = OBS_RUNS
    parameters['obs_result_path'] = os.path.join(opt.log_path, use_dataset,'vary_obs', model_name, str(parameters['time']))
    if not os.path.exists(parameters['obs_result_path']):
        os.makedirs(parameters['obs_result_path'])


parameters['dataset_path'] = os.path.join('datasets', use_dataset)


print('Model:', model_name)
print('Dataset:', use_dataset)

parameters['result_path'] = os.path.join(opt.log_path, use_dataset, model_name, str(parameters['time']))
if not os.path.exists(parameters['result_path']):
    os.makedirs(parameters['result_path'])

parameters['checkpoint_path'] = os.path.join(opt.log_path, use_dataset, model_name, parameters['checkpoint'])
if not os.path.exists(parameters['checkpoint_path']):
    os.makedirs(parameters['checkpoint_path'])  



# model specific parameters
if opt.model == 'LSTM':
    parameters['bidir'] = False

    if opt.dataset == 'braun':
        parameters['dpout'] = 0.5
        parameters['hdim'] = 100
        parameters['nepch'] = 25
        
        parameters['lrate'] = 0.001
        parameters['batch_size'] = 10
        # parameters['opsiz'] = 14

    elif opt.dataset == 'retail':
        parameters['dpout'] = 0.5
        parameters['hdim'] = 150
        parameters['nepch'] = 25
        
        parameters['lrate'] = 0.001
        parameters['batch_size'] = 10
        # parameters['opsiz'] = 54
    
    elif opt.dataset == 'travel':
        parameters['dpout'] = 0.5
        parameters['hdim'] = 100
        parameters['nepch'] = 25

        parameters['lrate'] = 0.001
        parameters['batch_size'] = 10
        # parameters['opsiz'] = 2

    elif opt.dataset == 'ubuntu':
        parameters['dpout'] = 0.5
        parameters['hdim'] = 100
        parameters['nepch'] = 25

        parameters['lrate'] = 0.001
        parameters['batch_size'] = 10
        # parameters['opsiz'] = 5 # TODO

    elif opt.dataset == 'webapp2':
        parameters['dpout'] = 0.5
        parameters['hdim'] = 100
        parameters['nepch'] = 25

        parameters['lrate'] = 0.001
        parameters['batch_size'] = 10
        # parameters['opsiz'] = 7 # TODO

elif opt.model == 'BiLSTM':
    parameters['bidir'] = True

    if opt.dataset == 'mareview':
        parameters['dpout'] = 0.5
        parameters['hdim'] = 200
        parameters['nepch'] = 5 # 10 for trec
        
        parameters['lrate'] = 0.001
        parameters['batch_size'] = 50
        # parameters['opsiz'] = 2 # 6 for trec2
    
    elif opt.dataset == 'braun':
        parameters['dpout'] = 0.5
        parameters['hdim'] = 150
        parameters['nepch'] = 25 # 10 for trec
        
        parameters['lrate'] = 0.001
        parameters['batch_size'] = 10
        # parameters['opsiz'] = 14 # 6 for trec2

    elif opt.dataset == 'retail':
        parameters['dpout'] = 0.5
        parameters['hdim'] = 150
        parameters['nepch'] = 25 # 10 for trec
        
        parameters['lrate'] = 0.001
        parameters['batch_size'] = 10
        # parameters['opsiz'] = 54 # 6 for trec2

    elif opt.dataset == 'travel':
        parameters['dpout'] = 0.5
        parameters['hdim'] = 100
        parameters['nepch'] = 25

        parameters['lrate'] = 0.001
        parameters['batch_size'] = 10
        # parameters['opsiz'] = 2

    elif opt.dataset == 'ubuntu':
        parameters['dpout'] = 0.5
        parameters['hdim'] = 100
        parameters['nepch'] = 25

        parameters['lrate'] = 0.001
        parameters['batch_size'] = 10
        # parameters['opsiz'] = 5 # TODO

    elif opt.dataset == 'webapp2':
        parameters['dpout'] = 0.5
        parameters['hdim'] = 100
        parameters['nepch'] = 25

        parameters['lrate'] = 0.001
        parameters['batch_size'] = 10
        # parameters['opsiz'] = 7 # TODO

elif opt.model == 'LSTM_BB':

    parameters['bidir'] = False
    parameters['sigmp'] = float(np.exp(-3))

    if opt.dataset == 'braun':
        parameters['dpout'] = 0.5
        parameters['hdim'] = 100
        parameters['nepch'] = 25 # 10 for trec
        
        parameters['lrate'] = 0.001
        parameters['batch_size'] = 10
        # parameters['opsiz'] = 14 # 6 for trec2

    elif opt.dataset == 'retail':
        parameters['dpout'] = 0.5
        parameters['hdim'] = 150
        parameters['nepch'] = 25 # 10 for trec
        
        parameters['lrate'] = 0.001
        parameters['batch_size'] = 10
        # parameters['opsiz'] = 54 # 6 for trec2

    elif opt.dataset == 'travel':
        parameters['dpout'] = 0.5
        parameters['hdim'] = 100
        parameters['nepch'] = 25

        parameters['lrate'] = 0.001
        parameters['batch_size'] = 10
        # parameters['opsiz'] = 2

    elif opt.dataset == 'ubuntu':
        parameters['dpout'] = 0.5
        parameters['hdim'] = 100
        parameters['nepch'] = 25

        parameters['lrate'] = 0.001
        parameters['batch_size'] = 10
        # parameters['opsiz'] = 5 # TODO

    elif opt.dataset == 'webapp2':
        parameters['dpout'] = 0.5
        parameters['hdim'] = 100
        parameters['nepch'] = 25

        parameters['lrate'] = 0.001
        parameters['batch_size'] = 10
        # parameters['opsiz'] = 7 # TODO

elif opt.model == 'BiLSTM_BB':
    parameters['bidir'] = True
    parameters['sigmp'] = float(np.exp(-3))

    if opt.dataset == 'mareview':
        parameters['dpout'] = 0.5
        parameters['hdim'] = 200
        parameters['nepch'] = 5 # 10 for trec
        
        parameters['lrate'] = 0.001
        parameters['batch_size'] = 50
        # parameters['opsiz'] = 2 # 6 for trec2
    
    elif opt.dataset == 'braun':
        parameters['dpout'] = 0.5
        parameters['hdim'] = 100
        parameters['nepch'] = 25 # 10 for trec
        
        parameters['lrate'] = 0.001
        parameters['batch_size'] = 10
        # parameters['opsiz'] = 14 # 6 for trec2

    elif opt.dataset == 'retail':
        parameters['dpout'] = 0.5
        parameters['hdim'] = 150
        parameters['nepch'] = 25 # 10 for trec
        
        parameters['lrate'] = 0.001
        parameters['batch_size'] = 10
        # parameters['opsiz'] = 54 # 6 for trec2

    elif opt.dataset == 'travel':
        parameters['dpout'] = 0.5
        parameters['hdim'] = 100
        parameters['nepch'] = 25

        parameters['lrate'] = 0.001
        parameters['batch_size'] = 10
        # parameters['opsiz'] = 2

    elif opt.dataset == 'ubuntu':
        parameters['dpout'] = 0.5
        parameters['hdim'] = 100
        parameters['nepch'] = 25

        parameters['lrate'] = 0.001
        parameters['batch_size'] = 10
        # parameters['opsiz'] = 5 # TODO

    elif opt.dataset == 'webapp2':
        parameters['dpout'] = 0.5
        parameters['hdim'] = 100
        parameters['nepch'] = 25

        parameters['lrate'] = 0.001
        parameters['batch_size'] = 10
        # parameters['opsiz'] = 7 # TODO

elif opt.model == 'NB':
    print(f'Model: {opt.model}, Dataset: {opt.dataset}')
elif opt.model == 'SVM':
    print(f'Model: {opt.model}, Dataset: {opt.dataset}')
else:
    raise NotImplementedError()


if parameters['crossval']:
    cross_val = Experiment(parameters)
    cross_val.save_param(parameters['cv_result_path'])
    # parameters = 
    best_w, best_h = cross_val.cv()
    parameters['worddim'] = best_w
    parameters['wordpath'] = WORD_PATHS[best_w]
    parameters['hdim'] = best_h
    # cross_val.create_meta()

# parse parameters again,
#   so they get saved in meta
#   and worddim and hdim get overwritten with best
if parameters['full']:
    print('beginning full experiment')
    full_data = Experiment(parameters)
    full_data.run()
    full_data.save_param()
    full_data.create_meta()

if parameters['intent']:
    intent = Experiment(parameters)
    intent.intent()
    intent.save_param(parameters['int_result_path'])

if parameters['vary_obs']:
    if parameters['dataset'] == 'retail':
        vary_obs = Experiment(parameters)
        vary_obs.vary_obs()
        vary_obs.save_param(parameters['obs_result_path'])
    else:
        raise NotImplementedError
    
print(f"Done, experiment took: {round(time()-parameters['time'], 2)} seconds")
# if opt.model == 'BiLSTM':
#     experiment.plot_loss()
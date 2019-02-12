# MScThesis
Master Thesis Econometrics - Intent Classification with Hierarchical and Bayesian Neural Nets

## How to use

## To-do's
A list of implementation that need to be done are in the `to-do.md`.

## Timeline
A timeline with deadlines and schedule for the to-do's are in `timeline.md`

## Folder structure
[link](https://stackoverflow.com/questions/9518646/tree-view-of-a-directory-folder-in-windows)
```
MScThesis/
|  README.md
|  LICENSE
|  losses.py
|  training.py (called by experiments.py, defines the training loop (looping over batches etc.), manages the optimizer, interacts with logging code)
|  experiments.py (called by train.py, parses config files, constructs dataset(s), models(s), passes them to training.py)
|  cls_scratch.ipynb (active)
|  train_cls.py (active)
|
|__bin/
|   train.py (command line interface for training models, creates experiment obj from experiments.py, change hyperparameters (incl data and model) via command line or config files)
|
|__wordvectors/
|  |__fasttext/
|  |__glove/
|  |__word2vec/
|
|__datasets/ (manages construction of datasets, handles data pipelining, staging areas, shuffling, reading raw binaries from disk, etc.)
|  |    data.py (msc2/code/models)
|  |    utils.py (msc2/code/models)
|  |__braun/
|  |__retain_bank/
|
|__models/ (model abstraction handles aspects of the model other than nn. E.g. input pre-processing, or output normalization)
|  |__naive_bayes/
|  |    naive_bayes.py (msc2/code/models)
|  |__svm/
|  |    svm.py (msc2/code/models)
|  |__neural_cls/ (active)
|     |  __init__.py
|     |
|     |__models/
|     |    __init__.py
|     |    bilstm.py
|     |    bilstm_bb.py
|     |
|     |__modules/
|     |    __init__.p
|     |    baseRNN.py
|     |    EncoderRNN.py
|     |    EncoderRNN_BB.py
|     |
|     |__util/
|          __init__.py
|          evaluator.py
|          initializer.py
|          loader.py
|          trainer.py
|          utils.py
|
|__networks (network abstr. manages creation of nn, DOES NOT HANDLE I-O/pre-post-preocessing of data, only computational graph creation)
|  |  base.py
|  |  cnn.py
|  |  resnet.py
|
|__configs/ (creation of model choice)
|     base_pose_estimation.yaml
|     pose_estimation_adaptation.yaml
|
|__notebooks/ (new)
|
|__plot/ (new, utils for plotting)
```
log the log-files to the user input --> such that you don't overwrite log with files from other settings.



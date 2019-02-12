import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F

import models.neural_cls
from models.neural_cls.util import Initializer
from models.neural_cls.util import Loader
from models.neural_cls.modules import EncoderRNN

class BiLSTM(nn.Module):
    
    def __init__(self, word_vocab_size, word_embedding_dim, word_hidden_dim, output_size, 
                 pretrained=None, n_layers = 1, bidirectional = True, dropout_p = 0.5, rnn_cell='lstm'):
        
        super(BiLSTM, self).__init__()
        
        self.word_vocab_size = word_vocab_size
        self.word_embedding_dim = word_embedding_dim
        self.word_hidden_dim = word_hidden_dim
        
        self.initializer = Initializer()
        self.loader = Loader()

        self.rnn_cell = rnn_cell
        
        self.word_encoder = EncoderRNN(word_vocab_size, word_embedding_dim, word_hidden_dim, 
                                       n_layers = n_layers, bidirectional = bidirectional,
                                       rnn_cell=rnn_cell)
        
        if pretrained is not None:
            self.word_encoder.embedding.weight = nn.Parameter(torch.FloatTensor(pretrained))
        
        self.dropout = nn.Dropout(p=dropout_p)
        
        hidden_size = 2*n_layers*word_hidden_dim if bidirectional and rnn_cell=='lstm' else n_layers*word_hidden_dim
        self.linear = nn.Linear(hidden_size, output_size)
        self.lossfunc = nn.CrossEntropyLoss()
        
    def forward(self, words, tags, wordslen): #, usecuda=True):
        
        batch_size, max_len = words.size()
        word_features = self.word_encoder(words, wordslen)
        word_features = self.dropout(word_features)
        output = self.linear(word_features)
        loss = self.lossfunc(output, tags)
        
        return loss
    
    def predict(self, words, wordslen, scoreonly=False, usecuda=True):
        
        batch_size, max_len = words.size()
        word_features = self.word_encoder(words, wordslen)
        word_features = self.dropout(word_features)
        output = self.linear(word_features)
        
        scores = torch.max(F.softmax(output, dim =1), dim=1)[0].data.cpu().numpy()
        if scoreonly:
            return scores
        
        prediction = torch.max(output, dim=1)[1].data.cpu().numpy().tolist()
        return scores, prediction
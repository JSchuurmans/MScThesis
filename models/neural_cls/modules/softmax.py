import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
# import data
import math
from models.neural_cls.util.utils import *

class LinearDecoder(nn.Module):
    def __init__(self, nhid, ntoken):
        super(LinearDecoder, self).__init__()
        self.decoder = nn.Linear(nhid, ntoken)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs):
        decoded = self.decoder(inputs.view(inputs.size(0)*inputs.size(1), inputs.size(2)))
        return decoded.view(inputs.size(0), inputs.size(1), decoded.size(1))


class HierarchicalSoftmax(nn.Module):
    def __init__(self, ntokens, nhid, tag_to_id, dataset='braun', ntokens_per_class = None):
        super(HierarchicalSoftmax, self).__init__()

        if dataset == 'braun':
            n_cat = 3
            n_subclasses = [2,5,8]
            # travel scheduling
            # ts = ['DepartureTime','FindConnection']
            # # AskUbuntu
            # au = ['Make Update', 'Setup Printer', 'Shutdown Computer',
            #         'Software Recommendation', 'None ask_ubuntu']
            # # webapp
            # wa = ['Change Password', 'Delete Account', 'Export Data', 'Filter Spam', 
            #         'Find Alternative', 'Sync Accounts', 'None web_app', 'Download Video']
            # li = [ts,au,wa]
        elif dataset == 'retail':
            n_cat = 7
            n_subclasses = [2,2,15,6,3,2,7]
        # Parameters
        self.ntokens = ntokens
        self.nhid = nhid

        if ntokens_per_class is None:
            ntokens_per_class = int(np.ceil(np.sqrt(ntokens)))

        self.ntokens_per_class = ntokens_per_class

        self.nclasses = int(np.ceil(self.ntokens * 1. / self.ntokens_per_class))
        self.ntokens_actual = self.nclasses * self.ntokens_per_class

        self.layer_top_W = nn.Parameter(torch.FloatTensor(self.nhid, self.nclasses), requires_grad=True)
        self.layer_top_b = nn.Parameter(torch.FloatTensor(self.nclasses), requires_grad=True)


        self.layer_bottom_W = nn.Parameter(torch.FloatTensor(self.nclasses, self.nhid, self.ntokens_per_class), requires_grad=True)
        self.layer_bottom_b = nn.Parameter(torch.FloatTensor(self.nclasses, self.ntokens_per_class), requires_grad=True)

        self.softmax = nn.Softmax(dim=1)

        self.init_weights()

    def init_weights(self):

        initrange = 0.1
        self.layer_top_W.data.uniform_(-initrange, initrange)
        self.layer_top_b.data.fill_(0)
        self.layer_bottom_W.data.uniform_(-initrange, initrange)
        self.layer_bottom_b.data.fill_(0)


    def forward(self, inputs, labels = None):

        batch_size, d = inputs.size()

        if labels is not None:

            label_position_top = labels / self.ntokens_per_class
            label_position_bottom = labels % self.ntokens_per_class

            layer_top_logits = torch.matmul(inputs, self.layer_top_W) + self.layer_top_b
            layer_top_probs = self.softmax(layer_top_logits)

            layer_bottom_logits = torch.squeeze(torch.bmm(torch.unsqueeze(inputs, dim=1), self.layer_bottom_W[label_position_top]), dim=1) + self.layer_bottom_b[label_position_top]
            layer_bottom_probs = self.softmax(layer_bottom_logits)

            # target_probs = layer_top_probs[torch.arange(batch_size).long(), label_position_top] * layer_bottom_probs[torch.arange(batch_size).long(), label_position_bottom]
            # cuda
            target_probs = layer_top_probs[torch.arange(batch_size).type(torch.cuda.LongTensor), label_position_top] * layer_bottom_probs[torch.arange(batch_size).type(torch.cuda.LongTensor), label_position_bottom]
            # print(target_probs)
            return target_probs

        else:
            
            
            # print(f'input size: {inputs.size()}')
            # Remain to be implemented
            layer_top_logits = torch.matmul(inputs, self.layer_top_W) + self.layer_top_b
            layer_top_probs = self.softmax(layer_top_logits)

            # print(self.softmax(torch.matmul(inputs, self.layer_bottom_W[0]) + self.layer_bottom_b[0]))
            # print(layer_top_probs[:,0])

            # # print(mul(layer_top_probs[:,0],tmp))
            # print(tmp)
            
            # word_probs = broad_ltp * tmp
            word_probs = layer_top_probs[:,0] * torch.t(self.softmax(torch.matmul(inputs, self.layer_bottom_W[0]) + self.layer_bottom_b[0])) # .transpose(0,1)
            # print(f'wordprob before concat: {word_probs.size()}')

            for i in range(1, self.nclasses):
                word_probs = torch.cat((word_probs, layer_top_probs[:,i] * torch.t(self.softmax(torch.matmul(inputs, self.layer_bottom_W[i]) + self.layer_bottom_b[i]))), dim=0)

            # print(f'after concat and t(): {torch.t(word_probs).size()}')
            return torch.t(word_probs)
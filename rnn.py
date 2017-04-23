import numpy as np
import pandas as pd
import pickle as pkl
import itertools
from sklearn import metrics
import datetime
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import random
import argparse
import time
import math
from torch.nn.utils import clip_grad_norm

import os
import sys

# For printing in real time on HPC
#sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

manualSeed = 1 # fix seed
print("Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='./data/', type=str, help='path to dataset')
parser.add_argument('--batchSize', type=int, default=20, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--weights', action='store_true', help='Weighted Cross Entropy')
parser.add_argument('--dropout', type=float, default=.0, help='Dropouts on discriminator')
parser.add_argument('--noise', type=float, default=.0, help='Add gaussian noise to real data')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
###
# RNN args
###
parser.add_argument('--model', type=str, default='LSTM', help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--optimizer', type=str, default='base', help='base | adam | RMSprop')
parser.add_argument('--layers', type=int, help='number of hidden layers', default=2)
parser.add_argument('--nhidden', type=int, help='hidden layers size', default=100)
parser.add_argument('--bptt', type=int, default=20,
                    help='sequence length')
parser.add_argument('--clip', type=float, default=0.5,
                    help='gradient clipping')
args = parser.parse_args()

inputSize = 217
log_interval = 100
eval_batch_size = 10

def loaderize(data_X, data_Y, balance, batch_size):
    # We want to bal
    tensor_data_set = torch.utils.data.TensorDataset(torch.from_numpy(data_X).float(), torch.from_numpy(data_Y))
    return torch.utils.data.DataLoader(tensor_data_set, batch_size=batch_size, shuffle=False, num_workers=int(args.workers))

###
# Load Data in tensors
###

trainloader = loaderize(np.load(args.dataroot+'train_X.pkl.npy'), np.load(args.dataroot+'train_y.pkl.npy'), False, args.batchSize * args.bptt)
valloader = loaderize(np.load(args.dataroot+'val_X.pkl.npy'), np.load(args.dataroot+'val_y.pkl.npy'), False, eval_batch_size * args.bptt)
testloader = loaderize(np.load(args.dataroot+'test_X.pkl.npy'), np.load(args.dataroot+'test_y.pkl.npy'), False, eval_batch_size * args.bptt)

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ninp, nhid, nlayers, dropout):
        super(RNNModel, self).__init__()
        # self.drop = nn.Dropout(dropout)
        if rnn_type == 'RNN_TANH':
            self.rnn = getattr(nn, 'RNN')(ninp, nhid, nlayers, nonlinearity='tanh', bias=False, dropout=dropout)
        elif rnn_type == 'RNN_RELU':
            self.rnn = getattr(nn, 'RNN')(ninp, nhid, nlayers, nonlinearity='relu', bias=False, dropout=dropout)
        else:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, bias=False, dropout=dropout)
        self.classifier = nn.Linear(nhid, 2)
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):
        input = self.dropout(input)
        output, hidden = self.rnn(input, hidden)
        output = self.dropout(output)
        return self.classifier(output.view(output.size(0)*output.size(1), output.size(2))), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

def clip_gradient(model, clip):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        modulenorm = p.grad.data.norm()
        totalnorm += modulenorm ** 2
    totalnorm = math.sqrt(totalnorm)
    return min(1, args.clip / (totalnorm + 1e-6))

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

model = RNNModel(args.model, inputSize, args.nhidden, args.layers, args.dropout)

input = torch.FloatTensor(args.bptt, args.batchSize, inputSize)
label = torch.LongTensor(args.batchSize * args.bptt)
if args.weights:
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1./.8, 1./.2]))
else:
    criterion = nn.CrossEntropyLoss()

input = Variable(input)
label = Variable(label)

if args.cuda:
    input = input.cuda()
    label = label.cuda()
    criterion.cuda()
    model.cuda()

def train(trainloader, epoch):
    model.train()
    hidden = model.init_hidden(args.batchSize)

    for i, (data, target) in enumerate(trainloader, 0):
        if data.size(0) < args.batchSize * args.bptt:
            smaller_batch_size= data.size(0) // args.bptt
            if smaller_batch_size == 0:
                break
            data = data[:smaller_batch_size * args.bptt]
            hidden = model.init_hidden(smaller_batch_size)
            data.resize_(args.bptt, smaller_batch_size, inputSize)
            target = target[:(smaller_batch_size * args.bptt)]
        else:
            data.resize_(args.bptt, args.batchSize, inputSize)

        input.data.resize_(data.size()).copy_(data)
        if args.noise > 0:
            noise = torch.FloatTensor(data.size()).normal_(0, args.noise)
            if args.cuda:
                noise = noise.cuda()
            input.data.add_(noise)
        label.data.resize_(target.size()).copy_(target)

        hidden = repackage_hidden(hidden)
        model.zero_grad()

        output, hidden = model(input, hidden)

        loss = criterion(output, label)
        loss.backward()

        if args.optimizer == 'base':
            clipped_lr = lr * clip_gradient(model, args.clip)
            for p in model.parameters():
                p.data.add_(-clipped_lr, p.grad.data)
        elif args.optimizer in ['adam', 'RMSprop']:
            clip_grad_norm(model.parameters(), args.clip)
            optimizer.step()

        if i % log_interval == 0:
            print('[%d/%d] [%d/%d] Train Loss : %.4f' %
                  (epoch, args.epochs,
                   i, len(trainloader),
                    loss.data[0]))

def test(testloader, epoch, isVal):
    model.eval()
    test_loss = 0
    correct = 0

    all_labels = 0
    all_preds = 0

    hidden = model.init_hidden(eval_batch_size)

    for i, (data, target) in enumerate(testloader, 0):
        if data.size(0) < args.batchSize * args.bptt:
            smaller_batch_size= data.size(0) // args.bptt
            if smaller_batch_size == 0:
                break
            data = data[:smaller_batch_size * args.bptt]
            hidden = model.init_hidden(smaller_batch_size)
            data.resize_(args.bptt, smaller_batch_size, inputSize)
            target = target[:(smaller_batch_size * args.bptt)]
        else:
            data.resize_(args.bptt, args.batchSize, inputSize)

        input.data.resize_(data.size()).copy_(data)
        label.data.resize_(target.size()).copy_(target)

        output, hidden = model(input, hidden)

        test_loss += criterion(output, label)
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(label.data).cpu().sum()
        if not torch.is_tensor(all_labels):
            all_labels = target
            all_preds = output.data[:,1]
        else:
            all_labels = torch.cat((all_labels, target), 0)
            all_preds = torch.cat((all_preds, output.data[:,1]), 0)

    test_loss /= len(testloader)

    auc = metrics.roc_auc_score(all_labels.cpu().numpy(), all_preds.cpu().numpy())
    if isVal:
        print('\n[%d/%d] ||VAL|| Average loss: %.4f, Accuracy: %d / %d (%.1f) AUC : %.6f \n' % (
                epoch, args.epochs,
                test_loss.data[0],
                correct, len(testloader.dataset), 100. * correct / len(testloader.dataset), auc)
             )
    else:
        print('\n[%d/%d] ||TEST|| Average loss: %.4f, Accuracy: %d / %d (%.1f) AUC : %.6f \n' % (
                epoch, epochs,
                test_loss.data[0],
                correct, len(testloader.dataset), 100. * correct / len(testloader.dataset), auc)
             )
    return test_loss.data[0]

# Loop over epochs.
lr = args.lr

if args.optimizer =='adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
elif args.optimizer =='RMSprop':
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

prev_val_loss = None

for epoch in range(1, args.epochs + 1):
    train(trainloader, epoch)
    val_loss = test(valloader, epoch, True)

    if prev_val_loss and (val_loss > prev_val_loss) and (epoch % 5 == 0):
        lr /= 4.0
        print('|||lr change||| %.4f' % (lr))
    prev_val_loss = val_loss
test(testloader, epoch, False)

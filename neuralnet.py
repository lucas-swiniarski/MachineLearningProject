import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--layers', type=int, help='number of hidden layers', default=2)
parser.add_argument('--nhidden', type=int, help='hidden layers size', default=100)
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--adam', action='store_true', help='Default RMSprop optimizer, wether to use Adam instead')
parser.add_argument('--dropout', type=float, default=.5, help='Dropouts on discriminator')
parser.add_argument('--noise', type=float, default=.1, help='Add gaussian noise to real data')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
args = parser.parse_args()

inputSize = 217
log_interval = 1000

def loaderize(data_X, data_Y, balance):
    # We want to bal
    tensor_data_set = torch.utils.data.TensorDataset(torch.from_numpy(data_X).float(), torch.from_numpy(data_Y))
    if balance:
        # We increase probability of minority class, and decrease probability of dominant class so in average
        # We sample the same amount of 1s and 0s even though classes are not balanced.
        proba_1 = data_Y.mean()
        weights = np.where(data_Y == 1., .5/proba_1, .5/(1.-proba_1))
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, data_Y.shape[0])
        return torch.utils.data.DataLoader(tensor_data_set, batch_size=args.batchSize, sampler=sampler, num_workers=int(args.workers))
    return torch.utils.data.DataLoader(tensor_data_set, batch_size=args.batchSize, shuffle=True, num_workers=int(args.workers))

###
# Load Data in tensors
###

trainloader = loaderize(pkl.load(open(args.dataroot+'train_X.pkl','rb')), pkl.load(open(args.dataroot+'train_y.pkl','rb')).values, True)
valloader = loaderize(pkl.load(open(args.dataroot+'val_X.pkl','rb')), pkl.load(open(args.dataroot+'val_y.pkl','rb')).values, False)
testloader = loaderize(pkl.load(open(args.dataroot+'test_X.pkl','rb')), pkl.load(open(args.dataroot+'test_y.pkl','rb')).values, False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(inputSize, args.nhidden)
        self.linears = nn.ModuleList([nn.Linear(args.nhidden, args.nhidden) for i in range(args.layers)])
        self.fc2 = nn.Linear(args.nhidden, 2)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, input):
        input = F.relu(self.fc1(input))
        input = self.dropout(input)
        for i, fc in enumerate(self.linears):
            input = self.dropout(F.relu(fc(input)))
        input = self.fc2(input)
        return input

model = Net()

if args.adam:
    optimizer = optim.Adam(model.parameters(), lr = args.lr, betas = (args.beta1, 0.999))
else:
    optimizer = optim.RMSprop(model.parameters(), lr = args.lr)

input = torch.FloatTensor(args.batchSize, inputSize)
label = torch.LongTensor(args.batchSize)
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

    for i, (data, target) in enumerate(trainloader, 0):
        input.data.resize_(data.size()).copy_(data)
        label.data.resize_(target.size()).copy_(target)
        model.zero_grad()
        output = model(input)
        loss = criterion(output, label)
        loss.backward()
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

    for i, (data, target) in enumerate(testloader, 0):
        input.data.resize_(data.size()).copy_(data)
        if args.noise > 0:
            # TODO@Lucas might also add noise at each layer of neural net.
            input.data.add_(torch.FloatTensor(data.size()).normal_(0, args.noise))
        label.data.resize_(target.size()).copy_(target)
        output = model(input)
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
    return test_loss

val_loss_stored = np.inf

for epoch in range(1, args.epochs + 1):
    train(trainloader, epoch)
    val_loss = test(valloader, epoch, True)
    if val_loss > val_loss_stored:
        args.lr /= 2
    val_loss_stored = val_loss
test(testloader, epoch, False)

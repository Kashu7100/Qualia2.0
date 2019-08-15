# -*- coding: utf-8 -*-
import qualia2 
from qualia2.core import *
from qualia2.functions import tanh, softmax_cross_entropy, transpose
from qualia2.nn import Module, GRU, Linear, Adadelta
from qualia2.data import FashionMNIST
from qualia2.util import Trainer
import matplotlib.pyplot as plt
import os
import argparse

path = os.path.dirname(os.path.abspath(__file__))

def data_trans(data):
    data = data.reshape(-1,28,28)
    return transpose(data, (1,0,2)).detach()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GRU example with Qualia2.0')
    parser.add_argument('mode', metavar='str', type=str, choices=['train', 'test'], help='select mode to run the model : train or test.')
    parser.add_argument('-i', '--itr', type=int, default=100, help=' Number of iterations for the training. Default: 100')
    parser.add_argument('-b', '--batch', type=int, default=100, help='Batch size to train the model. Default: 100')

    args = parser.parse_args()

    class GRU_classifier(Module):
        def __init__(self):
            super().__init__()
            self.gru = GRU(28,128,1)
            self.linear = Linear(128, 10)
            
        def forward(self, x, h0=qualia2.zeros((1,args.batch,128))):
            _, hx = self.gru(x, h0)
            out = self.linear(hx[-1])
            return out

    model = GRU_classifier()
    optim = Adadelta(model.params)
    mnist = FashionMNIST()

    trainer = Trainer(args.batch, path)
    trainer.set_data_transformer(data_trans)

    if args.mode == 'train':
        trainer.train(model, mnist, optim, softmax_cross_entropy, args.itr)

    if args.mode == 'test':
        trainer.test(model, mnist, args.batch, path+'/weights/gru')

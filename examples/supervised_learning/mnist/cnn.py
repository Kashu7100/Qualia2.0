# -*- coding: utf-8 -*-
import qualia2 
from qualia2.core import *
from qualia2.data import MNIST
from qualia2.nn.modules import Module, Conv2d, Linear
from qualia2.functions import leakyrelu, reshape, maxpool2d, mse_loss
from qualia2.nn.optim import Adadelta
from qualia2.util import Trainer
import matplotlib.pyplot as plt
import os
import argparse

path = os.path.dirname(os.path.abspath(__file__))

class CNN_classifier(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(1, 32, 5, padding=2)
        self.conv2 = Conv2d(32, 32, 5, padding=2)
        self.linear1 = Linear(32*7*7, 512)
        self.linear2 = Linear(512, 10)
    
    def forward(self, x):
        x = maxpool2d(leakyrelu(self.conv1(x)))
        x = maxpool2d(leakyrelu(self.conv2(x)))
        x = reshape(x, (-1, 32*7*7))
        x = leakyrelu(self.linear1(x))
        x = leakyrelu(self.linear2(x))
        return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dueling network example with Qualia2.0')
    parser.add_argument('mode', metavar='str', type=str, choices=['train', 'test'], help='select mode to run the model : train or test.')
    parser.add_argument('-i', '--itr', type=int, default=15, help=' Number of iterations for the training. Default: 100')
    parser.add_argument('-b', '--batch', type=int, default=100, help='Batch size to train the model. Default: 100')
    #parser.add_argument('-s', '--save', type=bool, default=False, help='Save mp4 video of the result. Default: False')
    #parser.add_argument('-p', '--plot', type=bool, default=False, help='Plot rewards over the training. Default: False')

    args = parser.parse_args()

    model = CNN_classifier()
    optim = Adadelta(model.params)
    mnist = MNIST()

    trainer = Trainer(args.batch, path)

    if args.mode == 'train':
        trainer.train(model, mnist, optim, mse_loss, args.itr)

    if args.mode == 'test':
        trainer.test(model, mnist, args.batch, path+'/weights/cnn')

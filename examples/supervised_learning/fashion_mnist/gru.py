# -*- coding: utf-8 -*-
import qualia2 
from qualia2.core import *
from qualia2.functions import tanh, softmax_cross_entropy, transpose, reshape
from qualia2.nn import Module, GRU, Linear, Adadelta
from qualia2.data import FashionMNIST
from qualia2.util import progressbar
from datetime import timedelta
import matplotlib.pyplot as plt
import time
import os

path = os.path.dirname(os.path.abspath(__file__))

# classification with reccurent network
class Reccurent(Module):
    def __init__(self):
        super().__init__()
        self.gru = GRU(28,128,1)
        self.linear = Linear(128, 10)
        
    def forward(self, x, h0):
        _, hx = self.gru(x, h0)
        out = self.linear(hx[-1])
        return out

model = Reccurent()
optim = Adadelta(model.params)
mnist = FashionMNIST()
mnist.batch = 100
h0 = qualia2.zeros((1,100,128))

epochs = 100

losses = []
start = time.time()
print('[*] training started.')
for epoch in range(epochs):
    for i, (data, label) in enumerate(mnist):
        data = reshape(data, (-1,28,28))
        data = transpose(data, (1,0,2)).detach()
        output = model(data, h0) 
        loss = softmax_cross_entropy(output, label)
        model.zero_grad()
        loss.backward()
        optim.step()
        progressbar(i, len(mnist), 'epoch: {}/{} test loss:{:.4f} '.format(epoch+1, epochs, to_cpu(loss.data) if gpu else loss.data), '(time: {})'.format(str(timedelta(seconds=time.time()-start))))
    losses.append(to_cpu(loss.data) if gpu else loss.data)
    if len(losses) > 1:
        if losses[-1] < losses[-2]:
            model.save(path+'/gru_test_{}'.format(epoch+1)) 
plt.plot([i for i in range(len(losses))], losses)
plt.show()
print('[*] training completed.')

print('[*] testing started.')
mnist.training = False
model.training = False
acc = 0
for i, (data, label) in enumerate(mnist): 
    data = reshape(data, (-1,28,28))
    data = transpose(data, (1,0,2))
    output = model(data, h0) 
    out = np.argmax(output.data, axis=1) 
    ans = np.argmax(label.data, axis=1)
    acc += sum(out == ans)/label.shape[0]
    progressbar(i, len(mnist))
print('\n[*] test acc: {:.2f}%'.format(float(acc/len(mnist)*100)))

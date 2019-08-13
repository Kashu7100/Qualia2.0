import qualia2
from qualia2.util import progressbar
from qualia2 import Tensor
from qualia2.nn import Linear, Module
from qualia2.functions import tanh,  mse_loss
from qualia2.nn.optim import Adadelta
from qualia2.core import *
from lorenz impot *
import os
import time
from datetime import timedelta
import random
import argparse

path = os.path.dirname(os.path.abspath(__file__))

class Model(Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(3, 256)
        self.linear2 = Linear(256, 3)
    
    def forward(self, s):
        s = tanh(self.linear1(s))
        return self.linear2(s)

# train the net with trapezoidal rule
# u_t = u_t1 + 1/2*dt*(f(u_t)+f(u_t1))
def train(model, optim, criteria, u, dt=0.01, epochs=2000):
    u_t1 = u[:-1]
    u_t = u[1:]
    start = time.time()
    for e in range(epochs):
        losses = []
        for b in range(len(u)//100):
            target = Tensor(2*(u_t[b*100:(b+1)*100] - u_t1[b*100:(b+1)*100]))
            output = dt*(model(Tensor(u_t[b*100:(b+1)*100])) + model(Tensor(u_t1[b*100:(b+1)*100])))
            loss = criteria(output, target)
            model.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.data)
        progressbar(e+1, epochs, 'loss: {}'.format(sum(losses)/len(losses)), '(time: {})'.format(str(timedelta(seconds=time.time()-start))))
    model.save(path+'/weights/lorenz')

if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description='Dueling network example with Qualia2.0')
    parser.add_argument('mode', metavar='str', type=str, choices=['train', 'test'], help='select mode to run the model : train or test.')
    parser.add_argument('-i', '--itr', type=int, default=200, help=' Number of epochs for the training. Default: 2000')
    
    model = Model()
    optim = Adadelta(model.params)
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(model, optim, mse_loss, u, dt, itr)
    if args.mode == 'test':
        model.load(path+'/weights/lorenz')

    def f(u, t):
        out = model(qualia2.array(u))
        return qualia2.to_cpu(out.data)
    
    learned_u = odeint(f, u0, t)
    plot3d(u, learned_u)

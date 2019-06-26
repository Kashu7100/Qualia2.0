# -*- coding: utf-8 -*-
import qualia2 
from qualia2.core import *
from qualia2.functions import sigmoid, tanh, concat, mse_loss, relu
from qualia2.nn import Module, Linear, BatchNorm1d, Conv2d, ConvTranspose2d, BatchNorm2d, Adam, Adadelta
from qualia2.data import MNIST
from qualia2.autograd import Tensor
from qualia2.util import progressbar
from datetime import timedelta
import matplotlib.pyplot as plt
import os
import time

path = os.path.dirname(os.path.abspath(__file__))

class Generator(Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(50, 128)
        self.linear2 = Linear(128, 256)
        self.linear3 = Linear(256, 512)
        self.linear4 = Linear(512, 784)

    def forward(self, x):
        x = tanh(self.linear1(x))
        x = tanh(self.linear2(x))
        x = tanh(self.linear3(x))
        x = relu(tanh(self.linear4(x)))
        return x

class Discriminator(Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(784, 512)
        self.linear2 = Linear(512, 256)
        self.linear3 = Linear(256, 128)
        self.linear4 = Linear(128, 1)

    def forward(self, x):
        x = tanh(self.linear1(x))
        x = tanh(self.linear2(x))
        x = tanh(self.linear3(x))
        x = sigmoid(self.linear4(x))
        return x
        
batch = 100
epochs = 200
z_dim = 50
smooth = 0.1

g = Generator()
d = Discriminator()

optim_g = Adam(g.params, 0.0004, (0.5, 0.999))
optim_d = Adam(d.params, 0.0002, (0.5, 0.999))

criteria = mse_loss

mnist = MNIST(flatten=True)
mnist.batch = batch

target_real = qualia2.ones((batch, 1))
target_fake = qualia2.zeros((batch,1))

check_noise = qualia2.randn(batch, z_dim)

start = time.time()
print('[*] training started.')
for epoch in range(epochs):
    for i, (data, _) in enumerate(mnist):  
        d.train()
        g.train()
        noise = qualia2.randn(batch, z_dim)
        fake_img = g(noise)
        # update Discriminator
        # feed fake images
        output_fake = d(fake_img.detach())
        loss_d_fake = criteria(output_fake, target_fake)
        # feed real images
        output_real = d(data)
        loss_d_real = criteria(output_real, target_real*(1-smooth))
        loss_d = loss_d_fake + loss_d_real    
        #if i%2 == 0 and np.sum(loss_d.data)/batch > 0.0015:
        d.zero_grad()
        loss_d.backward()
        optim_d.step()

        # update Generator
        d.eval()
        output = d(fake_img)
        loss_g = criteria(output, target_real)
        g.zero_grad()
        loss_g.backward()
        optim_g.step()
        
        progressbar(i, len(mnist), 'epoch: {}/{} loss_D:{:.4f} loss_G:{:.4f}'.format(epoch+1, epochs, to_cpu(np.sum(loss_d.data)/batch) if gpu else np.sum(loss_d.data)/batch, to_cpu(np.sum(loss_g.data)/batch) if gpu else np.sum(loss_g.data)/batch), '(time: {})'.format(str(timedelta(seconds=time.time()-start))))
    g.eval()
    
    fake_img = g(check_noise)
    for c in range(10):
        for r in range(10):
            plt.subplot(10,10,r+c*10+1)
            plt.xticks([]) 
            plt.yticks([]) 
            plt.grid(False)
            img = fake_img.data[r+c*10].reshape(28,28)
            plt.imshow(to_cpu(img) if gpu else img, cmap='gray', interpolation='nearest') 
    plt.savefig(path+'/gan/image_epoch{}.png'.format(epoch))      
    d.save(path+'/gan/d')   
    g.save(path+'/gan/g')  

print('[*] training completed.')

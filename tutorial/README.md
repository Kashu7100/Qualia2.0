## Table of Contents
| Component | Description |
| ---- | --- |
| [Automatic Differentiation](#automatic_differentiation) | usage of automatic differentiation with simple example |
| [Validation of Automatic Differentiation](#valid_automatic_differentiation) | numerical method to validate automatic differentiation |
| [Qualia Tensor](#qualia_tensor) | Tensor class in Qualia |
| [Network Definition](#network_definition) | create a custom neural network model with Qualia |
| [Model Summary](#model_summary) | get the summary of the neural network model |
| [Saving/Loading Weights](#save_load) | save and load the trained weights |
| [Setting up Optimizer](#optim_setup) | preparing optimizers to train a neural network |
| [Learning Qualia with Examples](#ex) | examples that cover essentials of Qualia |

<div id='automatic_differentiation'/>

## Automatic Differentiation
Fundamental of automatic differentiation [(AD)](https://en.wikipedia.org/wiki/Automatic_differentiation) is the decomposition of differentials based on the chain rule. Qualia implements the reverse accumulation AD in `qualia2.autograd`.

In the example code of this tutorial, we assume for simplicity that the following symbols are already imported.
```python
import qualia2
```
Qualia uses the so called “Define-by-Run” scheme, so forward computation itself defines the computational graph. By using a Tensor object, Qualia can keep track of every operation. Here, the resulting y is also a Tensor object, which points to its creator(s).
```python
x = qualia2.array([5])
y = x**2 - 2*x + 1
print(y)
# prints result of the computation: 
# [16] shape=(1,)
```
At this moment we can compute the derivative.
```python
y.backward()
print(x.grad)
# prints gradient of x:
# [8]
```
Note that this meets the result of symbolic differentiation.
<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\mathrm{d}&space;y}{\mathrm{d}&space;x}&space;=&space;2x&space;-2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\mathrm{d}&space;y}{\mathrm{d}&space;x}&space;=&space;2x&space;-2" title="\frac{\mathrm{d} y}{\mathrm{d} x} = 2x -2" /></a>
</p>
<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\therefore&space;y'(5)&space;=&space;8" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\therefore&space;y'(5)&space;=&space;8" title="\therefore y'(5) = 8" /></a>
</p>

All these computations can be generalized to a multidimensional tensor input. When the output is not a scalar quantity, a tenspr  with the same dimentions as the output that is filled with ones will be given by default to start backward computation.
```python
x = qualia2.array([[1, 2, 3], [4, 5, 6]])
y = x**2 - 2*x + 1
y.backward()
print(x.grad)
# prints gradients of x:
# [[ 0  2  4]
#  [ 6  8 10]]
```

With the autograd feature of Qualia, one can plot the derivative curve of a given function very easily. For instance, let function of interest be `y = x*sin(x)`.
```python
from qualia2.functions import *
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

x = qualia2.arange(-2*qualia2.pi,2*qualia2.pi,0.01)
y = x * sin(x)

y.backward()
if qualia2.gpu:
    ax.plot(qualia2.to_cpu(x.data), qualia2.to_cpu(y.data))
    ax.plot(qualia2.to_cpu(x.data), qualia2.to_cpu(x.grad))
else:
    ax.plot(x.data, y.data)
    ax.plot(x.data, x.grad)
ax.grid()

plt.show()
```
The following figure was obtained by the code above:
<p align="center">
  <img src="/assets/xsinx.png">
</p>

<div id='valid_automatic_differentiation'/>

## Validation of Automatic Differentiation 
One can use `util.check_function()` to validate the gradient caluclation of a function. `util.check_function()` internally calculates the gradient using numerical method and compares the result with automatic differentiation. 

```python
from qualia2.functions import *
from qualia2.util import check_function

check_function(sinc)
# [*] measured error:  6.662620763892326e-18
```

One can specify the domain to avoid null value for the function that has not defined region.

```python
check_function(tan, domain=(-np.pi/4, np.pi/4))
# [*] measured error:  1.0725402527904689e-12
```

<div id='qualia_tensor'/>

## Qualia Tensor
Every tensor calculation and automatic differentiation are done by the `Tensor` onject in Qualia. `Tensor` onject wraps `ndarray` objects along `creator` onject to perform automatic differentiation. A computational graph for a differentiation is defined dynamically as program runs. 

```python
x = qualia2.array([[1, 2, 3], [4, 5, 6]])
print(type(x))
# <class 'qualia2.autograd.Tensor'>
```

The gradient for a `Tensor` can be optionally replaced by a new gradient, which is additionally calculated by a hooked function.
```python
a = qualia2.rand(5,6)
a.backward()
print(a.grad)
# array([[1., 1., 1.],
#        [1., 1., 1.]])
```
If `lambda grad: 2*grad` is registered as a hook, the gradient will be doubled.
```python
a = qualia2.rand(5,6)
a.register_hook(lambda grad: 2*grad)
a.backward()
print(a.grad)
# array([[2., 2., 2.],
#        [2., 2., 2.]])
```

<div id='network_definition'/>

## Network Definition
In order to define a network, `nn.Module` needs to be inherited. Note that a user-defined model must have `super().__init__()` in the `__init__` of the model. 

```python
import qualia2
import qualia2.nn as nn
import qualia2.functions as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.maxpool2d(self.conv1(x), (2,2)))
        x = F.relu(F.maxpool2d(self.conv2(x), (2,2)))
        x = F.reshape(x,(-1, 500))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```
If the model is sequential, there is another option to use `nn.Sequential`.

<div id='model_summary'/>

## Model Summary
Having a visualization of the model is very helpful while debugging your network. You can obtain a network summary by `your_model.summary(input_shape)`. Note that the `input_size` is required to make a forward pass through the network.

```python
import qualia2
import qualia2.nn as nn
import qualia2.functions as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.maxpool2d(self.conv1(x), (2,2)))
        x = F.relu(F.maxpool2d(self.conv2(x), (2,2)))
        x = F.reshape(x,(-1, 500))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Model()
model.summary((1, 1, 28, 28))
```
following is the output: 
```
------------------------------------------------------------------
                            Model: Net
------------------------------------------------------------------
| layers    |    input shape     |    output shape    | params # |
==================================================================
| Conv2d-0  |   (1, 1, 28, 28)   |  (1, 10, 26, 26)   |   260    |
| Conv2d-1  |  (1, 10, 13, 13)   |  (1, 20, 11, 11)   |   5020   |
| Linear-2  |      (1, 500)      |      (1, 50)       |  25050   |
| Linear-3  |      (1, 50)       |      (1, 10)       |   510    |
==================================================================
total params: 30840
training mode: True
------------------------------------------------------------------
```


<div id='save_load'/>

## Saving/Loading a Trained Weights
In order to save the trained weights of a model, one can simply use `Module.save(filename)` method. The weights are saved in [hdf5](https://support.hdfgroup.org/HDF5/whatishdf5.html) format. To load the saved weights, use `Module.load(filename)` method.
```python
import os
path = os.path.dirname(os.path.abspath(__file__)

# assume model has been defined
model.save(path+'/weights')
model.load(path+'/weights')
```

<div id='optim_setup'/>

## Setting up Optimizer
Optimizers require the model parameters. Put `Module.params` as the first argument for the optimizer. Other arguments such as learning rate are optional. 
```python
optim = Optimizer(model.params)
```

<div id='ex'/>

## Leaning with Examples
- [Example with Spiral Dataset - Decision Boundary](#ex1)
- [Example with MNIST Dataset - PCA](#ex2)
- [Example with FashionMNIST Dataset - Classification wirh GRU](#ex3)
- [Example with Lorenz System - Regression](#ex4)
- [Example with CartPole Env - DQN](#ex5)
- [Example with MountainCar Env - Dueling Network](#ex6)
- [Example with BipedalWalker Env - TD3](#ex7)

<div id='ex1'/>

### Example with Spiral Dataset - Decision Boundary
Neural networks can be viewed as a universal approximation function. Let's use a simple dataset called Spiral to see how neural net can obtain a non-linear decision boundary. To visualize the dataset, one can use the `Spiral.show()` method as follows: 

```python
from qualia2.data.basic import Spiral

data = Spiral()
data.show()
```
<p align="center">
  <img src="/assets/spiral.png">
</p>

The following network was used for the Spiral dataset. One can use `Spiral.plot_decision_boundary(model)` to visualize the decision boundary attained by the model.

```python
import qualia2
from qualia2.data.basic import Spiral
from qualia2.nn.modules import Module, Linear
from qualia2.functions import sigmoid, mse_loss
from qualia2.nn.optim import Adadelta
import matplotlib.pyplot as plt

class Net(Module):
    def __init__(self):
        super().__init__()
        self.l1 = Linear(2, 15)
        self.l2 = Linear(15, 3)

    def forward(self, x):
        x = sigmoid(self.l1(x))
        x = sigmoid(self.l2(x))
        return x
        
net = Net()
optim = Adadelta(net.params)

data = Spiral()
data.batch = 100

losses=[]

for _ in range(3000):
    for feature, target in data:
        out = net(feature)
        loss = mse_loss(out, target)
        losses.append(qualia2.to_cpu(loss.data))
        optim.zero_grad()
        loss.backward()
        optim.step()

plt.plot([i for i in range(len(losses))], losses)
plt.show()

data.plot_decision_boundary(net)
```
The following plot show the change in loss over iterations.
<p align="center">
  <img src="/assets/spiral_loss.png">
</p>
Following is the decision boundary obtained. We can observe that the network could fit the non-linear dataset.
<p align="center">
  <img src="/assets/spiral_boundary.png">
</p>

<div id='ex2'/>

### Example with MNIST Dataset - PCA
Neural networks can be used in dimensionality reduction (PCA) since the internal state of the hourglass neural networks can be regarded as the lower dimensional representation of the input. Let's use MNIST dataset.

```python
from qualia2.data.basic import MNIST

data = MNIST()
data.show()
```
<p align="center">
  <img src="/assets/mnist_data.png">
</p>

```python
from qualia2.core import *
from qualia2.data import MNIST
from qualia2.nn.modules import Module, Conv2d, Linear
from qualia2.functions import leakyrelu, reshape, maxpool2d, mse_loss
from qualia2.nn.optim import Adadelta
from qualia2.util import trainer, tester
import os.path
path = os.path.dirname(os.path.abspath(__file__))

class Classifier(Module):
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

mnist = MNIST()
model = Classifier()
optim = Adadelta(model.params)
trainer(model, mse_loss, optim, mnist, 10, 100, path+'/qualia2_mnist')
tester(model, mnist, 50, path+'/qualia2_mnist')
```
With the CUDA acceleration, this simple model can achieve more than 97% accuracy on the testing data in tens of minutes. Then we utilize this network to conduct the principal component analysis.

```python
class PCA(Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(10, 32)
        self.linear2 = Linear(32, 2)
        self.linear3 = Linear(2, 32)
        self.linear4 = Linear(32, 10)
        
    def forward(self, x):
        if self.training:
            x = tanh(self.linear1(x))
            x = tanh(self.linear2(x))
            x = tanh(self.linear3(x))
            x = self.linear4(x)
            return x
        else:
            x = tanh(self.linear1(x))
            x = tanh(self.linear2(x))
            return x

model1 = Classifier()
model1.training = False
model2 = PCA()
optim = Adadelta(model2.params)

losses = []

for i in range(10):
    for feature, _ in mnist:
        tmp = model1(feature)
        out = model2(tmp.detach())
        loss = mse_loss(out, tmp)
        losses.append(qualia2.to_cpu(loss.data))
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(losses[-1])
```
With the following code, we will visualize the result. Note that since we used tanh as the activation function, we can set the limits of the figure to [-1, 1]. 
```python
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
fig, ax = plt.subplots()
plt.xlim(-1,1)
plt.ylim(-1,1)

for feature, _ in mnist:
    tmp = model1(feature)
    out = model2(tmp)
    img = OffsetImage(qualia2.to_cpu(1-feature.data[0].reshape(28,28)), cmap='gray', interpolation='nearest', zoom=0.5)
    ab = AnnotationBbox(img, (qualia2.to_cpu(out.data[0,0]), qualia2.to_cpu(out.data[0,1])), frameon=False) 
    ax.add_artist(ab)
plt.show()
```
The figure below suggests that the internal state of the neural network distinguishes the handwritten digits to some extent. Interestingly, the digits with similar portions, such as '8' and '3,' are tend to be closer each other.
<p align="center">
  <img src="/assets/mnist_map_colored.png">
</p>

<div id='ex3'/>

### Example with FashionMNIST - Classification with GRU
RNNs are often utilized for language model or time series prediction; however, they can also be used for image recongnition tasks. We will demonstrate the classification task on FashionMNIST with GRU as an example. Below is the visualization for the dataset.

```python
from qualia2.data.basic import FashionMNIST

data = FashionMNIST()
data.show()
```
<p align="center">
  <img src="/assets/fashion_mnist_data.png">
</p>

According to [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist), here are some reasons that FashionMNIST is preffered over MNIST for testing classifier:   	

> **MNIST is too easy.** Convolutional nets can achieve 99.7% on MNIST. Classic machine learning algorithms can also achieve 97% easily.
 	
> **MNIST is overused.** In this April 2017 Twitter thread, Google Brain research scientist and deep learning expert Ian Goodfellow calls for people to move away from MNIST.
 	
> **MNIST can not represent modern CV tasks**, as noted in this April 2017 Twitter thread, deep learning expert/Keras author François Chollet.

The following model assumes that final hidden states of the GRU embraces the input features. The model can be also implemented using the last output from GRU; however, backpropagation is much slower if implemented in this manner.

```python
from qualia2.functions import tanh, softmax_cross_entropy, transpose, reshape
from qualia2.nn import Module, GRU, Linear, Adadelta
from qualia2.util import progressbar
from datetime import timedelta
import matplotlib.pyplot as plt
import time
import os

path = os.path.dirname(os.path.abspath(__file__))

class Reccurent(Module):
    def __init__(self):
        super().__init__()
        self.gru = GRU(28,128,1)
        self.linear = Linear(128, 10)
        
    def forward(self, x, h0):
        _, hx = self.gru(x, h0)
        out = self.linear(hx[-1])
        """ same but slower
        out, _ = self.gru(x, h0)
        out = self.linear(out[-1])
        """
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
```
The following plot show the change in loss over epochs.

<p align="center">
  <img src="/assets/fashion_mnist_gru_loss.png">
</p>

With this model, `89.95%` accuracy on test dataset was achieved.

<div id='ex4'/>

### Example with Lorenz system - Regression
To explore the identification of chaotic dynamics evolving on a finite dimensional attractor, let's consider the nonlinear Lorenz system:
<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\dot{x}&space;=&space;10(y-x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dot{x}&space;=&space;10(y-x)" title="\dot{x} = 10(y-x)" /></a>
</p>
<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\dot{y}&space;=&space;x(28-z)-y" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dot{y}&space;=&space;x(28-z)-y" title="\dot{y} = x(28-z)-y" /></a>
</p>
<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\dot{z}&space;=&space;xy-(8/3)z" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dot{z}&space;=&space;xy-(8/3)z" title="\dot{z} = xy-(8/3)z" /></a>
</p>

Here is the code for Lorenz system simulation:
```python
from scipy.integrate import odeint
import numpy as np

def lorenz(u, t):
    x, y, z = u
    sigma = 10.0
    beta = 8.0/3.0
    rho = 28.0
        
    dxdt = sigma*(y-x)
    dydt = x*(rho-z)-y
    dzdt = x*y-beta*z
    return np.array([dxdt,dydt,dzdt])

dt = 0.01   
t = np.arange(0,25, dt)
u0 = np.array([-8.0, 7.0, 27])
u = odeint(lorenz, u0, t)
```

The trapezoidal rule is a numerical method to solve ordinary differential equations that approximates solutions to initial value problems of the form: 

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\dot{y}&space;=&space;f(t,y)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dot{y}&space;=&space;f(t,y)" title="\dot{y} = f(t,y)" /></a>
</p>
<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=y(t_0)=y_0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y(t_0)=y_0" title="y(t_0)=y_0" /></a>
</p>
The trapezoidal rule states that:
<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=y_n&space;-&space;y_n_-_1&space;=&space;\int_{t_n_-_1}^{t_n}f(t,y)dt&space;\approx&space;\Delta&space;t(f(t_n,y_n)&plus;f(t_n_-_1,y_n_-_1))/2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_n&space;-&space;y_n_-_1&space;=&space;\int_{t_n_-_1}^{t_n}f(t,y)dt&space;\approx&space;\Delta&space;t(f(t_n,y_n)&plus;f(t_n_-_1,y_n_-_1))/2" title="y_n - y_n_-_1 = \int_{t_n_-_1}^{t_n}f(t,y)dt \approx \Delta t(f(t_n,y_n)+f(t_n_-_1,y_n_-_1))/2" /></a>
</p>
The model will be trained so that the trapezoidal rule is satisfied. LHS will be the target and the RHS will be the sum of the outputs from the model multiplied by a time step.


<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=2(y_n&space;-&space;y_n_-_1)&space;=&space;\Delta&space;t(f(t_n,y_n)&plus;f(t_n_-_1,y_n_-_1))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?2(y_n&space;-&space;y_n_-_1)&space;=&space;\Delta&space;t(f(t_n,y_n)&plus;f(t_n_-_1,y_n_-_1))" title="2(y_n - y_n_-_1) = \Delta t(f(t_n,y_n)+f(t_n_-_1,y_n_-_1))" /></a>
</p>

```python
import qualia2
from qualia2.util import progressbar
from qualia2 import Tensor
from qualia2.nn import Linear, Module
from qualia2.functions import tanh,  mse_loss
from qualia2.nn.optim import Adadelta
from qualia2.core import *
import os
import time
from datetime import timedelta
import random

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
    model.save(path+'/lorenz')

model = Model()
optim = Adadelta(model.params)
train(model, optim, mse_loss, u, dt)

def f(u, t):
    out = model(qualia2.array(u))
    return qualia2.to_cpu(out.data)
    
learned_u = odeint(f, u0, t)
```

Following is the obtained result:
<p align="center">
  <img src="/assets/lorenz_compare.png">
</p>

<div id='ex5'/>

### Example with Cart-Pole - DQN
Q-learning updates the action value according to the following equation:

<p align="center">
  <img src="/assets/q-learning.PNG"/>
</p>

When the learning converges, the second term of the equation above approaches to zero.
Note that when the policy that never takes some of the pairs of state and action, the action value function for the pair will never be learned, and learning will not properly converge. DQN is Q-Learning with a deep neural network as a Q function approximator. DQN learns to minimize the loss of the following function, where E indicates loss function:

<p align="center">
  <img src="/assets/dqn.PNG"/>
</p>

DQN updates the parameters θ according to the following gradient:

<p align="center">
  <img src="/assets/dqn_grad.PNG"/>
</p>

Qualia2 provides `DQN` (`DQNTrainer`) class and `Env` class for handy testing of DQN. As an example, let's use [CartPole](https://gym.openai.com/envs/CartPole-v1/) task from Gym. One can visualize the environment with `Env.show()` method.
```python
from qualia2.rl.envs import CartPole
from qualia2.rl import ReplayMemory
from qualia2.rl.agents import DQNTrainer, DQN
from qualia2.nn.modules import Module, Linear
from qualia2.functions import tanh
from qualia2.nn.optim import Adadelta

class Model(Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(4, 32)
        self.linear2 = Linear(32, 32)
        self.linear3 = Linear(32, 2)

    def forward(self, x):
        x = tanh(self.linear1(x))
        x = tanh(self.linear2(x))
        x = tanh(self.linear3(x))
        return x

env = CartPole()
env.show()
```

<p align="center">
  <img src="/assets/cartpole_random.gif">
</p>

In order to execute experience replay to train the neural network, simply use `Trainer.train()` method.
```python
agent = DQN.init(env, Model())
agent.set_optim(Adadelta)
trainer = DDQNTrainer(ReplayMemory, batch=80, capacity=10000)
agent = trainer.train(env, agent, episodes=200, filename=path+'/dqn_cartpole')
trainer.plot()
agent.play(env)
```
Within 50 episodes in this case, the model could achive 200 steps. Try running the code several times in case of not achieving 200 steps in 50 episodes since the leaning depends on the initial weights of the network.
```bash
...
[*] Episode: 28 - steps: 200 loss: 0.003701 reward: 1.0
[*] Episode: 29 - steps: 110 loss: 0.005517 reward: -1.0
[*] Episode: 30 - steps: 200 loss: 0.003471 reward: 1.0
[*] Episode: 31 - steps: 200 loss: 0.003282 reward: 1.0
[*] Episode: 32 - steps: 200 loss: 0.00372 reward: 1.0
[*] Episode: 33 - steps: 200 loss: 0.003556 reward: 1.0
[*] Episode: 34 - steps: 200 loss: 0.002909 reward: 1.0
[*] Episode: 35 - steps: 200 loss: 0.00297 reward: 1.0
[*] Episode: 36 - steps: 200 loss: 0.00303 reward: 1.0
[*] Episode: 37 - steps: 200 loss: 0.003275 reward: 1.0
[*] Episode: 38 - steps: 200 loss: 0.002473 reward: 1.0
[*] Episode: 39 - steps: 200 loss: 0.002647 reward: 1.0
[*] Episode: 40 - steps: 200 loss: 0.002749 reward: 1.0
[*] Episode: 41 - steps: 200 loss: 0.002741 reward: 1.0
...
```
Following is the animated result:
<p align="center">
  <img src="/assets/cartpole_dqn.gif">
</p>

find more in: [example page](https://github.com/Kashu7100/Qualia2.0/tree/master/examples/reinforcement_learning/inverted_pendulum)

<div id='ex6'/>

### Example with Mountain Car - Dueling Network
The information within a Q function can be divided into two: a part determined mostly by state; and a part influenced by an action choosed. Dueling network separates the Q function into Value, a part that is determined by state, and Advantage, a part that is influenced by the action. This enables the model to learn the parameters that is related to Value every step regardless of action choosed, i.e. the model can learn faster than DQN.

<p align="center">
  <img src="/assets/dueling_net.PNG"/>
</p>

As an example, let's use [MountainCar](https://gym.openai.com/envs/MountainCar-v0/) task from Gym.
```python
# -*- coding: utf-8 -*- 
from qualia2.environment.mountaincar import MountainCar
from qualia2.applications.dqn import DDQN
from qualia2.nn.modules import Module, Linear
from qualia2.functions import tanh, mean
from qualia2.nn.optim import Adadelta
import os
path = os.path.dirname(os.path.abspath(__file__))

class Model(Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(2, 64)
        self.linear2 = Linear(64, 64)
        self.linear_adv = Linear(64, 3)
        self.linear_val = Linear(64, 1)

    def forward(self, x):
        x = tanh(self.linear1(x))
        x = tanh(self.linear2(x))
        adv = self.linear_adv(x)
        val = self.linear_val(x)
        # subtract mean of adv to cancel out the effect of bias
        result = val + adv - mean(adv, axis=1).reshape(-1,1) 
        return result

agent = DDQN(Network, Adadelta, 10000, 80)
env = MountainCar(agent, 200, 300)
env.show()
```

<p align="center">
  <img src="/assets/mountaincar_random.gif">
</p>

In order to execute experience replay to train the model, use `Environment.run()` method.
```python
env.run()
env.plot_rewards()
env.animate(path+'/mountaincar')
```
<p align="center">
  <img src="/assets/mountaincar_loss.png">
</p>
Following is the animated result:
<p align="center">
  <img src="/assets/mountaincar_duelingnet.gif">
</p>

<div id='ex7'/>

## Example with BipedalWalker Env - TD3
writing...

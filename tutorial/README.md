## Table of Contents
- [Automatic Differentiation](#automatic_differentiation)
- [Validation of Automatic Differentiation](#valid_automatic_differentiation)
- [Network Definition](#network_definition)
- [Model Summary](#model_summary)
- [Example with Spiral Dataset - Decision Boundary](#ex1)
- [Example with MNIST Dataset - PCA](#ex2)
- [Example with Cart-Pole - DQN](#ex3)

<div id='automatic_differentiation'/>

## Automatic Differentiation
Fundamental of automatic differentiation [(AD)](https://en.wikipedia.org/wiki/Automatic_differentiation) is the decomposition of differentials based on the chain rule. Qualia implements the reverse accumulation AD in qualia2.autograd.

In the example code of this tutorial, we assume for simplicity that the following symbols are already imported.
```python
import qualia2
```
Qualia uses the so called “Define-by-Run” scheme, so forward computation itself defines the computational graph. By using a Tensor object, Qualia can keep track of every operation. Here, the resulting y is also a Tensor object, which points to its creator(s).
```python
x = qualia2.array([5])
y = x**2 - 2*x + 1
# prints result of the computation: 
# [16] shape=(1,)
print(y)
```
At this moment we can compute the derivative.
```python
y.backward()
# prints gradient of x:
# [8]
print(x.grad)
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
# prints gradients of x:
# [[ 0  2  4]
#  [ 6  8 10]]
print(x.grad)
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
One can use `util.check_function()` to validate the gradient caluclation of a function.

```python
from qualia2.functions import *
from qualia2.util import check_function

check_function(sinc)
#[*] measured error:  6.662620763892326e-18
```

<div id='network_definition'/>

## Network Definition
In order to define a network, one needs to inherit `nn.Module`.

```python
import qualia2
import qualia2.nn as nn
import qualia2.functions as F

class Net(nn.Module):
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

class Net(nn.Module):
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

model = Net()
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

<div id='ex1'/>

## Example with Spiral Dataset - Decision Boundary
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

## Example with MNIST Dataset - PCA
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
With the CUDA acceleration, this simple model can achieve about 97% accuracy on the testing data in tens of minutes. Then we utilize this network to conduct the principal component analysis.

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
        tmp.creator = None
        out = model2(tmp)
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

## Example with Cart-Pole - DQN
DQN is Q-Learning with a Deep Neural Network as a function approximator. Qualia2 provides `DQN` class and `Environment` class for handy testing for DQN. As an example, let's use [CartPole](https://gym.openai.com/envs/CartPole-v1/) task from Gym. One can visualize the environment with `Environment.show()` method.
```python
from qualia2.environment.cartpole import CartPole
from qualia2.applications.dqn import DQN
from qualia2.nn.modules import Module, Linear
from qualia2.functions import tanh, sigmoid
from qualia2.nn.optim import Adadelta
import os
path = os.path.dirname(os.path.abspath(__file__))

class Network(Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(4, 100)
        self.linear2 = Linear(100, 100)
        self.linear3 = Linear(100, 2)

    def forward(self, x):
        x = tanh(self.linear1(x))
        x = tanh(self.linear2(x))
        x = sigmoid(self.linear3(x))
        return x

agent = DQN(Network, Adadelta, 10000, 50)
env = CartPole(agent, 200, 50)
env.show()
```

<p align="center">
  <img src="/assets/cartpole_random.gif">
</p>

In order to execute experience replay to train the neural network, simply use `Environment.run()` method.
```
env.run()
env.animate(path+'/dqn_cartpole')
```
Within 50 episodes in this case, the model could achive 200 steps. Try running the code several times in case of not achieving 200 steps in 50 episodes since the leaning depends on the initial weights of the network.
```bash
...
[*] episode 34: finished after 172 steps
[*] episode 35: finished after 108 steps
[*] episode 36: finished after 200 steps
[*] episode 37: finished after 200 steps
[*] episode 38: finished after 200 steps
[*] episode 39: finished after 200 steps
[*] episode 40: finished after 200 steps
[*] episode 41: finished after 200 steps
[*] episode 42: finished after 200 steps
[*] episode 43: finished after 200 steps
[*] episode 44: finished after 200 steps
[*] episode 45: finished after 200 steps
...
```
Following is the animated result:
<p align="center">
  <img src="/assets/cartpole_dqn.gif">
</p>

<div id='ex4'/>

## Example with Mountain Car Continuous - A2C

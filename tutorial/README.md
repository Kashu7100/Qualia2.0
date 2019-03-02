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

## Validation of Automatic Differentiation 
One can use `util.check_function()` to validate the gradient caluclation of a function.

```python
from qualia2.functions import *
from qualia2.util import check_function

check_function(sinc)
#[*] measured error:  6.662620763892326e-18
```

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
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = reshape(x,(-1, 320))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```
If the model is sequential, there is another option to use `nn.Sequential`.

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
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = reshape(x,(-1, 320))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Net()
model.summary((1, 1, 28, 28))
```

```bash

```


## Model Summary
Having a visualization of the model is very helpful while debugging your network. You can obtain a network summary by `your_model.summary(input_shape)`. Note that the `input_size` is required to make a forward pass through the network.

```python
import qualia2
import qualia2.nn as nn
import qualia2.functions as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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

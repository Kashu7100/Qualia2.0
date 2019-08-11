from qualia2.data.basic import Spiral
from qualia2.nn.modules import Module, Linear
from qualia2.functions import sigmoid, mse_loss
from qualia2.nn.optim import Adadelta
from qualia2.util import Trainer
import matplotlib.pyplot as plt

class Model(Module):
    def __init__(self):
        super().__init__()
        self.l1 = Linear(2, 15)
        self.l2 = Linear(15, 3)

    def forward(self, x):
        x = sigmoid(self.l1(x))
        x = sigmoid(self.l2(x))
        return x
    
model = Model()
optim = Adadelta(model.params)

trainer = Trainer(batch=100, path=path)
trainer.train(model, data, optim, mse_loss, epochs=3000)

data.show_decision_boundary(model)

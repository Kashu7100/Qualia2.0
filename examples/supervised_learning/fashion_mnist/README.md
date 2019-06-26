# FashionMNIST
Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. 
<p align="center">
  <img src="/assets/fashion_mnist_data.png">
</p>

## Classification with GRU
### Usage
Following are the commands used to train and test the model:

To train the model:
```bash
python gru.py train --itr 200 --batch 100
```

To run with pre-trained weights:
```bash
python gru.py test 
```
### Results
Loss plot:
<p align="center">
  <img src="/assets/fashion_mnist_gru_loss.png">
</p>

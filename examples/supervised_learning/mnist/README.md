# MNIST
The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.

<p align="center">
  <img src="/assets/mnist_data.png">
</p>

## Classification with CNN
### Details
We will use a small CNN to perform the classification.
```bash
----------------------------------------------------------------------------
                           Model: CNN_classifier                            
----------------------------------------------------------------------------
| layers              |    input shape     |    output shape    | params # |
============================================================================
| Conv2d-0            |   (1, 1, 28, 28)   |  (1, 32, 28, 28)   |   832    |
| Conv2d-1            |  (1, 32, 14, 14)   |  (1, 32, 14, 14)   |  25632   |
| Linear-2            |     (1, 1568)      |      (1, 512)      |  803328  |
| Linear-3            |      (1, 512)      |      (1, 10)       |   5130   |
============================================================================
total params: 834922
```

### Usage
Following are the commands used to train and test the model:

To train the model:
```bash
python cnn.py train --itr 15 --batch 100
```

To run with pre-trained weights:
```bash
python cnn.py test 
```
### Results
<p align="center">
  <img src="/assets/mnist_cnn_loss.png">
</p>

```bash
[*] test acc: 97.26%
```

## Classification with GRU
### Details
The GRU model takes rows of the image as shown assuming the hidden state of GRU will contain a context of digits. 
<p align="center">
  <img src="/assets/mnist-gru.gif" width="200">
</p>

### Usage
Following are the commands used to train and test the model:

To train the model:
```bash
python gru.py train --itr 15 --batch 100
```

To run with pre-trained weights:
```bash
python gru.py test 
```
### Results
<p align="center">
  <img src="/assets/mnist_gru_loss.png">
</p>

```bash
[*] test acc: 99.06%
```

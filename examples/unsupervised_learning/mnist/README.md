# MNIST
The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.

<p align="center">
  <img src="/assets/mnist_data.png">
</p>

## GAN
### Usage
Following are the commands used to train and test the model:

To train the model:
```bash
python gan.py train --itr 200 --z_dim 50 --batch 100
```

To run with pre-trained weights:
```bash
python gan.py test 
```

You can also experience the effect of z values on generated image with the jupyter notebook.
```bash
jupyter notebook gan_interactive.ipynb
```

### Results
The obtained result:
<p align="center">
  <img src="/assets/gan_mnist.gif">
</p>

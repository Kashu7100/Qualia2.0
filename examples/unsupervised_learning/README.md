# Generative adversarial networks
## [GAN](/examples/unsupervised_learning/mnist)
GANs - Generative Adversarial networks - were introduced by Ian Goodfellow in 2014. GANs can create new data by competing two neural networks: *generator* <a href="https://www.codecogs.com/eqnedit.php?latex=G:&space;Z\rightarrow&space;X" target="_blank"><img src="https://latex.codecogs.com/gif.latex?G:&space;Z\rightarrow&space;X" title="G: Z\rightarrow X" /></a> and *discriminator* <a href="https://www.codecogs.com/eqnedit.php?latex=D:&space;X\rightarrow&space;\mathbb{R}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?D:&space;X\rightarrow&space;\mathbb{R}" title="D: X\rightarrow \mathbb{R}" /></a>. 

<p align="center">
  <img src="/assets/gan_diagram.PNG"/>
  <br>
  <b> Fig.1: </b> GAN architecture
</p>

<p align="center">
  <img src="/assets/gan.PNG"/>
</p>

## DCGAN
DCGAN - Deep Convolutional GANs - is updated version of GANs presented in 2015. It utilizes convolutional layers along batch normalization and leaky ReLU.

<p align="center">
  <img src="/assets/DCGAN.png"/ width=600>
  <br>
  <b> Fig.2: </b> DCGAN generator used for LSUN scene modeling
</p>

## LSGAN
GANs had problem that learning is unstable. LSGAN - Least Square GANs, which employs least square loss instead of binary cross entropy loss for its loss function was introduced intended to improve the stability of the learning. 

<p align="center">
  <img src="/assets/lsgan.PNG"/>
</p>

## Conditional GAN
GANs or DCGAN could not specify the class when generating image, since the generator totally depends on the random noise. In conditional GAN, generator gets noise as well as a label as an input; the desired output is to be the one corresponds to the label.  Also, discriminator gets the label along the real or generated image.

<p align="center">
  <img src="/assets/conditional_gan.PNG"/>
  <br>
  <b> Fig.3: </b> Conditional GAN architecture
</p>

# Dimentionality reduction

# Autoencoders

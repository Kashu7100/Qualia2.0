## GAN
GANs - Generative Adversarial networks - were introduced by Ian Goodfellow in 2014. GANs can create new data by competing two neural networks: *generator* (G) and *discriminator* (D). 

<p align="center">
  <img src="/assets/gan_diagram.PNG"/>
  <img src="/assets/gan.PNG"/>
</p>

## DCGAN
DCGAN - Deep Convolutional GANs - is updated version of GANs presented in 2015. It utilizes convolutional layers along batch normalization and leaky ReLU.

<p align="center">
  <img src="/assets/DCGAN.png"/>
  <br>
  <b> Figure: </b> DCGAN generator used for LSUN scene modeling (retrieved from <a href="https://arxiv.org/abs/1511.06434">Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks</a>)
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
</p>

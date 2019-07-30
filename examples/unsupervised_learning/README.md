# Generative adversarial networks
## GAN
[[paper]](https://arxiv.org/pdf/1406.2661.pdf) [[example]](/examples/unsupervised_learning/mnist)

GANs - Generative Adversarial networks - were introduced by Ian Goodfellow in 2014. GANs can create new data by competing two neural networks: *generator* <a href="https://www.codecogs.com/eqnedit.php?latex=G:&space;Z\rightarrow&space;X" target="_blank"><img src="https://latex.codecogs.com/gif.latex?G:&space;Z\rightarrow&space;X" title="G: Z\rightarrow X" /></a> and *discriminator* <a href="https://www.codecogs.com/eqnedit.php?latex=D:&space;X\rightarrow&space;\mathbb{R}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?D:&space;X\rightarrow&space;\mathbb{R}" title="D: X\rightarrow \mathbb{R}" /></a>. 

<p align="center">
  <img src="/assets/gan_diagram.PNG"/>
  <br>
  <b> Fig.1: </b> GAN architecture
</p>

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\underset{G}{min}\:&space;\underset{D}{max}&space;\:&space;V_D(D,G)=\mathbb{E}_{x\sim&space;P(x)}\{log(D(x)))\}&plus;\mathbb{E}_{z\sim&space;P(z)}\{log(1-D(G(z)))\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\underset{G}{min}\:&space;\underset{D}{max}&space;\:&space;V_D(D,G)=\mathbb{E}_{x\sim&space;P(x)}\{log(D(x)))\}&plus;\mathbb{E}_{z\sim&space;P(z)}\{log(1-D(G(z)))\}" title="\underset{G}{min}\: \underset{D}{max} \: V_D(D,G)=\mathbb{E}_{x\sim P(x)}\{log(D(x)))\}+\mathbb{E}_{z\sim P(z)}\{log(1-D(G(z)))\}" /></a>
</p>

## Conditional GAN
[[paper]](https://arxiv.org/pdf/1411.1784)

GANs or DCGAN could not specify the class when generating image, since the generator totally depends on the random noise. In conditional GAN, generator gets noise as well as a label as an input; the desired output is to be the one corresponds to the label.  Also, discriminator gets the label along the real or generated image.

<p align="center">
  <img src="/assets/conditional_gan.PNG"/>
  <br>
  <b> Fig.3: </b> Conditional GAN architecture
</p>

## DCGAN
[[paper]](https://arxiv.org/pdf/1511.06434.pdf)

DCGAN - Deep Convolutional GANs - is updated version of GANs presented in 2015. It utilizes convolutional layers along with batch normalization and leaky ReLU.

<p align="center">
  <img src="/assets/DCGAN.png"/ width=600>
  <br>
  <b> Fig.2: </b> DCGAN generator used for LSUN scene modeling
</p>

## LSGAN
[[paper]](https://arxiv.org/pdf/1611.04076.pdf)

GANs had problem that learning is unstable. LSGAN - Least Square GANs, which employs least square loss instead of binary cross entropy loss for its loss function was introduced intended to improve the stability of the learning. 

<p align="center">
  <a href="https://www.codecogs.com/eqnedit.php?latex=\underset{D}{min}\:&space;V_D(D,G)=\frac{1}{2}\mathbb{E}_{x\sim&space;P(x)}\{(D(x)-1)^2&space;\}&plus;\frac{1}{2}\mathbb{E}_{z\sim&space;P(z)}\{D(G(z))^2&space;\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\underset{D}{min}\:&space;V_D(D,G)=\frac{1}{2}\mathbb{E}_{x\sim&space;P(x)}\{(D(x)-1)^2&space;\}&plus;\frac{1}{2}\mathbb{E}_{z\sim&space;P(z)}\{D(G(z))^2&space;\}" title="\underset{D}{min}\: V_D(D,G)=\frac{1}{2}\mathbb{E}_{x\sim P(x)}\{(D(x)-1)^2 \}+\frac{1}{2}\mathbb{E}_{z\sim P(z)}\{D(G(z))^2 \}" /></a>
</p>

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\underset{G}{min}\:&space;V_G(D,G)=\frac{1}{2}\mathbb{E}_{z\sim&space;P(z)}\{(D(G(z))-1)^2&space;\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\underset{G}{min}\:&space;V_G(D,G)=\frac{1}{2}\mathbb{E}_{z\sim&space;P(z)}\{(D(G(z))-1)^2&space;\}" title="\underset{G}{min}\: V_G(D,G)=\frac{1}{2}\mathbb{E}_{z\sim P(z)}\{(D(G(z))-1)^2 \}" /></a>
</p>

## CycleGAN
[[paper]](https://arxiv.org/pdf/1703.10593.pdf)

Image-to-image translation is a class of vision and graphics problems where the goal is to learn the mapping between an input image and an output image using a training set of aligned image pairs.

# Dimentionality reduction

# Autoencoders

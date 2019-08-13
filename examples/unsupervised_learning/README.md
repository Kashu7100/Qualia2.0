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

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\underset{G}{min}\:&space;\underset{D}{max}&space;\:&space;V_D(D,G)=\mathbb{E}_{x\sim&space;P(x)}\{log(D(x,l)))\}&plus;\mathbb{E}_{z\sim&space;P(z)}\{log(1-D(G(z,l)))\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\underset{G}{min}\:&space;\underset{D}{max}&space;\:&space;V_D(D,G)=\mathbb{E}_{x\sim&space;P(x)}\{log(D(x,l)))\}&plus;\mathbb{E}_{z\sim&space;P(z)}\{log(1-D(G(z,l)))\}" title="\underset{G}{min}\: \underset{D}{max} \: V_D(D,G)=\mathbb{E}_{x\sim P(x)}\{log(D(x,l)))\}+\mathbb{E}_{z\sim P(z)}\{log(1-D(G(z,l)))\}" /></a>
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

# Autoencoders
## Data-driven Discovery of Nonlinear Dynamical Systems

[[paper]](https://arxiv.org/pdf/1801.01236.pdf) [[example]](examples/unsupervised_learning/lorenz_system)

In the following equation, <a href="https://www.codecogs.com/eqnedit.php?latex=x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x" title="x" /></a> denotes the state of the system at time <a href="https://www.codecogs.com/eqnedit.php?latex=t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?t" title="t" /></a> and the function <a href="https://www.codecogs.com/eqnedit.php?latex=f" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f" title="f" /></a> describes the evolution of the system. The term <a href="https://www.codecogs.com/eqnedit.php?latex=u" target="_blank"><img src="https://latex.codecogs.com/gif.latex?u" title="u" /></a> can be the external forcing or feedback control. The goal is to determine the function <a href="https://www.codecogs.com/eqnedit.php?latex=f" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f" title="f" /></a> and consequently discover the underlying dynamical system from data.  
<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\dot{x}&space;=&space;f(x(t),u(t),t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dot{x}&space;=&space;f(x(t),u(t),t)" title="\dot{x} = f(x(t),u(t),t)" /></a>
</p>

By applying the general form of a linear multistep method with <a href="https://www.codecogs.com/eqnedit.php?latex=M" target="_blank"><img src="https://latex.codecogs.com/gif.latex?M" title="M" /></a> steps to the equation above, given the measurements of the state <a href="https://www.codecogs.com/eqnedit.php?latex=x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x" title="x" /></a> from <a href="https://www.codecogs.com/eqnedit.php?latex=t&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?t&space;=&space;0" title="t = 0" /></a> to <a href="https://www.codecogs.com/eqnedit.php?latex=t&space;=&space;N" target="_blank"><img src="https://latex.codecogs.com/gif.latex?t&space;=&space;N" title="t = N" /></a>, the following equation can be obtained:

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\sum_{m=0}^{M}[\alpha_mx_{n-m}&plus;\Delta&space;t\beta_m&space;f(x_{n-m},u_{n-m},t_{n-m})]=0,&space;\;&space;n=M,...,N" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sum_{m=0}^{M}[\alpha_mx_{n-m}&plus;\Delta&space;t\beta_m&space;f(x_{n-m},u_{n-m},t_{n-m})]=0,&space;\;&space;n=M,...,N" title="\sum_{m=0}^{M}[\alpha_mx_{n-m}+\Delta t\beta_m f(x_{n-m},u_{n-m},t_{n-m})]=0, \; n=M,...,N" /></a>
</p>

The function <a href="https://www.codecogs.com/eqnedit.php?latex=f" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f" title="f" /></a> is apploximated by a neural netwok. The neural network is trained so that the LHS of the equation above approaches to zero.

When <a href="https://www.codecogs.com/eqnedit.php?latex=M=1,&space;\alpha_0=-1,&space;\alpha_1=1,&space;and&space;\,&space;\beta_0=\beta_1=1/2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?M=1,&space;\alpha_0=-1,&space;\alpha_1=1,&space;and&space;\,&space;\beta_0=\beta_1=1/2" title="M=1, \alpha_0=-1, \alpha_1=1, and \, \beta_0=\beta_1=1/2" /></a>, the equation states the trapezoidal rule.

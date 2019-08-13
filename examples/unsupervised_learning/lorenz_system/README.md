# Lorenz system

To explore the identification of chaotic dynamics evolving on a finite dimensional attractor, let's consider the nonlinear Lorenz system:
<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\dot{x}&space;=&space;10(y-x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dot{x}&space;=&space;10(y-x)" title="\dot{x} = 10(y-x)" /></a>
</p>
<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\dot{y}&space;=&space;x(28-z)-y" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dot{y}&space;=&space;x(28-z)-y" title="\dot{y} = x(28-z)-y" /></a>
</p>
<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\dot{z}&space;=&space;xy-(8/3)z" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dot{z}&space;=&space;xy-(8/3)z" title="\dot{z} = xy-(8/3)z" /></a>
</p>

## Details

The trapezoidal rule is a numerical method to solve ordinary differential equations that approximates solutions to initial value problems of the form: 

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\dot{y}&space;=&space;f(t,y)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dot{y}&space;=&space;f(t,y)" title="\dot{y} = f(t,y)" /></a>
</p>
<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=y(t_0)=y_0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y(t_0)=y_0" title="y(t_0)=y_0" /></a>
</p>
The trapezoidal rule states that:
<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=y_n&space;-&space;y_n_-_1&space;=&space;\int_{t_n_-_1}^{t_n}f(t,y)dt&space;\approx&space;\Delta&space;t(f(t_n,y_n)&plus;f(t_n_-_1,y_n_-_1))/2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_n&space;-&space;y_n_-_1&space;=&space;\int_{t_n_-_1}^{t_n}f(t,y)dt&space;\approx&space;\Delta&space;t(f(t_n,y_n)&plus;f(t_n_-_1,y_n_-_1))/2" title="y_n - y_n_-_1 = \int_{t_n_-_1}^{t_n}f(t,y)dt \approx \Delta t(f(t_n,y_n)+f(t_n_-_1,y_n_-_1))/2" /></a>
</p>
The model will be trained so that the trapezoidal rule is satisfied. LHS will be the target and the RHS will be the sum of the outputs from the model multiplied by a time step.


<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=2(y_n&space;-&space;y_n_-_1)&space;=&space;\Delta&space;t(f(t_n,y_n)&plus;f(t_n_-_1,y_n_-_1))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?2(y_n&space;-&space;y_n_-_1)&space;=&space;\Delta&space;t(f(t_n,y_n)&plus;f(t_n_-_1,y_n_-_1))" title="2(y_n - y_n_-_1) = \Delta t(f(t_n,y_n)+f(t_n_-_1,y_n_-_1))" /></a>
</p>

## Usage
Following are the commands used to train and test the model:

To train the model:
```bash
$ python trapezoidal.py train --itr 2000
```
To test the model:
```bash
$ python trapezoidal.py test
```

## Result
Following is the obtained result:
<p align="center">
  <img src="/assets/lorenz_compare.png">
</p>

## Introduction
Animals, including humans, change their behavior through experience. It is said that the brain has three types of leaning system: supervised learning, reinforcement learning, and unsupervised leaning.

<p align="center">
  <img src="/assets/Brain_DL.PNG"/>
  <br>
  <b> Fig.1: </b> Learning scheme in the brain.
</p>

## [Supervised Learning](/examples/supervised_learning) 
*Supervised Learning* is a machine learning technique that expects a model to learn the input-to-label mapping of data where an input and a label associated with that input are given.

## [Unsupervised Learning](/examples/unsupervised_learning)
*Unsupervised learning* is a machine learning technique that expects a model to learn patterns in the input data. Unsupervised learning such as Hebbian learning or self-organization has been heavily utilized by the living creatures. In general, unsupervised system is better than supervised system in finding new patterns or features in the inputs.

In 1949, Donald O. Hebb argued that: 
> "When an axon of cell A is near enough to excite a cell B and repeatedly or persistently takes part in firing it, some growth process or metabolic change takes place in one or both cells such that A's efficiency, as one of the cells firing B, is increased." 

This rule is called Hebbian learning; and this synaptic plasticity is thought to be the basic phenomenon in our learning and memory.

## [Reinforcement Learning](/examples/reinforcement_learning)
*Reinforcement Learning* is a machine learning technique that enables an agent to learn in an interactive environment by trial and error using feedback from its own actions and experiences assuming Markov Decision Process (MDP). Reinforcement Learning named after operant conditioning, a method of learning that occurs through rewards and punishments for behavior, presented by B. F. Skinner.

<p align="center">
  <img src="/assets/reinforcement-learning.jpg" width="450"/>
  <br>
  <b> Fig.2: </b> Learning scheme for reinforcement learning assuming MDP.
</p>  

### Markov property
A stochastic process has the Markov property if the conditional probability distribution of future states of the process depends only upon the present state. That is, the state s(t+1) and reward r(t+1) at time t+1 depends on the present state s(t) and the action a(t). 

### Value function 
A value function <a href="https://www.codecogs.com/eqnedit.php?latex=V^\pi(s)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?V^\pi(s)" title="V^\pi(s)" /></a> under a policy <a href="https://www.codecogs.com/eqnedit.php?latex=\pi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\pi" title="\pi" /></a> is the expectation value of the total discounted reward at given state s.

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=V^\pi&space;(s)=&space;\mathbb{E}^\pi&space;\{G_t&space;|&space;S_t=s\}&space;=&space;\mathbb{E}^\pi&space;\{\sum_{\tau=0}^{\infty}&space;\gamma&space;^\tau&space;R_t_&plus;_\tau_&plus;_1|&space;S_t=s\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?V^\pi&space;(s)=&space;\mathbb{E}^\pi&space;\{G_t&space;|&space;S_t=s\}&space;=&space;\mathbb{E}^\pi&space;\{\sum_{\tau=0}^{\infty}&space;\gamma&space;^\tau&space;R_t_&plus;_\tau_&plus;_1|&space;S_t=s\}" title="V^\pi (s)= \mathbb{E}^\pi \{G_t | S_t=s\} = \mathbb{E}^\pi \{\sum_{\tau=0}^{\infty} \gamma ^\tau R_t_+_\tau_+_1| S_t=s\}" /></a>
</p>  

Similarly, the expectation value of the total discounted reward at given state s and an action a is represented by <a href="https://www.codecogs.com/eqnedit.php?latex=Q^\pi(s,a)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Q^\pi(s,a)" title="Q^\pi(s,a)" /></a>.

<a href="https://www.codecogs.com/eqnedit.php?latex=Q^\pi&space;(s,a)=&space;\mathbb{E}^\pi&space;\{G_t&space;|&space;S_t=s,&space;A_t=a\}&space;=&space;\mathbb{E}^\pi&space;\{\sum_{\tau=0}^{\infty}&space;\gamma&space;^\tau&space;R_t_&plus;_\tau_&plus;_1|&space;S_t=s,&space;A_t=a\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Q^\pi&space;(s,a)=&space;\mathbb{E}^\pi&space;\{G_t&space;|&space;S_t=s,&space;A_t=a\}&space;=&space;\mathbb{E}^\pi&space;\{\sum_{\tau=0}^{\infty}&space;\gamma&space;^\tau&space;R_t_&plus;_\tau_&plus;_1|&space;S_t=s,&space;A_t=a\}" title="Q^\pi (s,a)= \mathbb{E}^\pi \{G_t | S_t=s, A_t=a\} = \mathbb{E}^\pi \{\sum_{\tau=0}^{\infty} \gamma ^\tau R_t_+_\tau_+_1| S_t=s, A_t=a\}" /></a>

### TD error 

<p align="center">
  <img src="/assets/TD_error.png" width="500"/>
  <br>
  <b> Fig.3: </b> Firing of dopamine neurons and its correspondence with the TD error [1,2]. 
</p>  
In the first case, an unpredicted reward (R) occurs, and a burst of dopamine firing follows. In the second case, a predicted reward occurs, and a burst follows the onset of the predictor (CS or conditioned stimulus), but there is no firing after the predicted reward. In the bottom case, a predicted reward is omitted, with a corresponding trough in dopamine responding.

---
[1] Schultx, W., et al. (1997) Predictive Reward Signal of Dopamine Neurons Science 275: 1593-1599 

[2] Doya K. (2007). Reinforcement learning: Computational theory and biological mechanisms. HFSP journal, 1(1), 30–40. doi:10.2976/1.2732246/10.2976/1

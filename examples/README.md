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
  <img src="/assets/reinforcement-learning.jpg"/>
  <br>
  <b> Fig.2: </b> Learning scheme for reinforcement learning assuming MDP.
</p>  

### Markov property
A stochastic process has the Markov property if the conditional probability distribution of future states of the process depends only upon the present state. That is, the state s(t+1) and reward r(t+1) at time t+1 depends on the present state s(t) and the action a(t). 

<p align="center">
  <img src="/assets/TD_error.png"/>
  <br>
  <b> Fig.3: </b> Response of dopamine neurons [1]. 
</p>  

---
[1] Schultx, W., et al. (1997) Predictive Reward Signal of Dopamine Neurons Science 275: 1593-1599 

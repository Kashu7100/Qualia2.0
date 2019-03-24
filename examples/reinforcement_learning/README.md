<p align="center">
  <img src="/assets/reinforcement_learning_algorithm_map.png"/>
</p>

## Q-learning
Q-learning updates the action value according to the following equation:

<p align="center">
  <img src="/assets/q-learning.PNG"/>
</p>

When the learning converges, the second term of the equation above approaches to zero.
Note that when the policy that never takes some of the pairs of state and action, the action value function for the pair will never be learned, and learning will not properly converge. 

## DQN
DQN is Q-Learning with a deep neural network as a Q function approximator. DQN learns to minimize the loss of the following function, where E indicates loss function:

<p align="center">
  <img src="/assets/dqn.PNG"/>
</p>

## DDQN

## Dueling Network

## A2C

## TD3

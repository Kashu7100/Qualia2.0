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

DQN updates the parameters θ according to the following gradient:

<p align="center">
  <img src="/assets/dqn_grad.PNG"/>
</p>

## DDQN
It is known that it will degrade the learning stability if the target Q value is calculated with the same parameters θ as the updating parameters. In order to stablize the learning process, two networks, main network Qm and target network Qt, are introduced. The updating equation will be:

<p align="center">
  <img src="/assets/ddqn.PNG"/>
</p>

## Dueling Network

## A2C

## TD3

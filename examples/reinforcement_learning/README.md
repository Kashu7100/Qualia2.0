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

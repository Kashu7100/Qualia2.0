<p align="center">
  <img src="/assets/reinforcement_learning_algorithm_map.png"/>
</p>

# Value Based
## Q-learning
Q-learning updates the action value according to the following equation:

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=Q(S_t,A_t)&space;\leftarrow&space;Q(S_t,A_t)&plus;\alpha&space;(R_{t&plus;1}&plus;\gamma&space;\,&space;\underset{a'\in&space;A}{max}\:&space;Q(S_{t&plus;1},a')-Q(S_t,A_t))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Q(S_t,A_t)&space;\leftarrow&space;Q(S_t,A_t)&plus;\alpha&space;(R_{t&plus;1}&plus;\gamma&space;\,&space;\underset{a'\in&space;A}{max}\:&space;Q(S_{t&plus;1},a')-Q(S_t,A_t))" title="Q(S_t,A_t) \leftarrow Q(S_t,A_t)+\alpha (R_{t+1}+\gamma \, \underset{a'\in A}{max}\: Q(S_{t+1},a')-Q(S_t,A_t))" /></a>
</p>

When the learning converges, the second term of the equation above approaches to zero.
Note that when the policy that never takes some of the pairs of state and action, the action value function for the pair will never be learned, and learning will not properly converge. 

## [DQN](/examples/reinforcement_learning/inverted_pendulum)
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

## [Dueling Network](/examples/reinforcement_learning/mountain_car)
The information within a Q function can be divided into two: a part determined mostly by state; and a part influenced by an action choosed. Dueling network separates the Q function into Value, a part that is determined by state, and Advantage, a part that is influenced by the action. This enables the model to learn the parameters that is related to Value every step regardless of action choosed, i.e. the model can learn faster than DQN.

<p align="center">
  <img src="/assets/dueling_net.PNG"/>
</p>

## Rainbow
Rainbow commbines DQN with six extensions (the number of colors in a rainbow!) that address the limitaions of the original DQN algorithm.
1. ### DQN
2. ### DDQN
3. ### Prioritized experience replay
4. ### Dueling network
5. ### Multi-step learning
6. ### Distributional RL
7. ### Noisy network

# Actor-Critic
## A2C
A2C stands for Advantage Actor-Critic. 

## DDPG
Deep Deterministic Policy Gradient (DDPG) is an off-policy, model-free, and actor-critic algorithm. 

## [TD3](/examples/reinforcement_learning/bipedal_walker)
Twin Delayed DDPG (TD3). DDPG is frequently brittle with respect to hyperparameters and tunings. A common failure mode for DDPG is that the learned Q-function begins to dramatically overestimate Q-values, which then leads to the policy breaking, exploiting the errors in the Q-function.

# Policy Based
## TRPO
Trust Region Policy Optimization (TRPO)


## PPO
Proximal Policy Optimization (PPO) is a policy gradient based method. 

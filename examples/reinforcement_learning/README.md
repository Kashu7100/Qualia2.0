<p align="center">
  <img src="/assets/reinforcement_learning_algorithm_map.png"/>
</p>

# Value Based
## SARSA
Sarsa updates the action value according to the following equation:
<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=Q(S_t,A_t)&space;\leftarrow&space;Q(S_t,A_t)&plus;\alpha&space;(R_{t&plus;1}&plus;\gamma&space;\,&space;Q(S_{t&plus;1},A_{t&plus;1})-Q(S_t,A_t))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Q(S_t,A_t)&space;\leftarrow&space;Q(S_t,A_t)&plus;\alpha&space;(R_{t&plus;1}&plus;\gamma&space;\,&space;Q(S_{t&plus;1},A_{t&plus;1})-Q(S_t,A_t))" title="Q(S_t,A_t) \leftarrow Q(S_t,A_t)+\alpha (R_{t+1}+\gamma \, Q(S_{t+1},A_{t+1})-Q(S_t,A_t))" /></a>
</p>

## Q-learning
Q-learning updates the action value according to the following equation:

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=Q(S_t,A_t)&space;\leftarrow&space;Q(S_t,A_t)&plus;\alpha&space;(R_{t&plus;1}&plus;\gamma&space;\,&space;\underset{a'\in&space;A}{max}\:&space;Q(S_{t&plus;1},a')-Q(S_t,A_t))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Q(S_t,A_t)&space;\leftarrow&space;Q(S_t,A_t)&plus;\alpha&space;(R_{t&plus;1}&plus;\gamma&space;\,&space;\underset{a'\in&space;A}{max}\:&space;Q(S_{t&plus;1},a')-Q(S_t,A_t))" title="Q(S_t,A_t) \leftarrow Q(S_t,A_t)+\alpha (R_{t+1}+\gamma \, \underset{a'\in A}{max}\: Q(S_{t+1},a')-Q(S_t,A_t))" /></a>
</p>

When the learning converges, the second term of the equation above approaches to zero.
Note that when the policy that never takes some of the pairs of state and action, the action value function for the pair will never be learned, and learning will not properly converge. 

## [DQN](/examples/reinforcement_learning/inverted_pendulum)
DQN is Q-Learning with a deep neural network as a Q function approximator. DQN learns to minimize the TD error with some evaluation function <a href="https://www.codecogs.com/eqnedit.php?latex=J" target="_blank"><img src="https://latex.codecogs.com/gif.latex?J" title="J" /></a>. 

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=J(R_{t&plus;1}&plus;\gamma&space;\,&space;\underset{a'\in&space;A}{max}\:&space;Q_\theta(S_{t&plus;1},a'),Q_\theta(S_t,A_t))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?J(R_{t&plus;1}&plus;\gamma&space;\,&space;\underset{a'\in&space;A}{max}\:&space;Q_\theta(S_{t&plus;1},a'),Q_\theta(S_t,A_t))" title="J(R_{t+1}+\gamma \, \underset{a'\in A}{max}\: Q_\theta(S_{t+1},a'),Q_\theta(S_t,A_t))" /></a>
</p>

Generally, mean square error is used as evaluation function.

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=J(\theta)&space;=&space;\frac{1}{2}&space;(R_{t&plus;1}&plus;\gamma&space;\,&space;\underset{a'\in&space;A}{max}\:&space;Q_\theta(S_{t&plus;1},a')-Q_\theta(S_t,A_t))^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?J(\theta)&space;=&space;\frac{1}{2}&space;(R_{t&plus;1}&plus;\gamma&space;\,&space;\underset{a'\in&space;A}{max}\:&space;Q_\theta(S_{t&plus;1},a')-Q_\theta(S_t,A_t))^2" title="J(\theta) = \frac{1}{2} (R_{t+1}+\gamma \, \underset{a'\in A}{max}\: Q_\theta(S_{t+1},a')-Q_\theta(S_t,A_t))^2" /></a>
</p>

### neural fitted Q
It is known that it will degrade the learning stability if the target Q value is calculated with the same parameters <a href="https://www.codecogs.com/eqnedit.php?latex=\theta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta" title="\theta" /></a> as the updating parameters. In order to stablize the learning process, the parameter <a href="https://www.codecogs.com/eqnedit.php?latex=\theta^-" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta^-" title="\theta^-" /></a>, which is a periodic copy of the <a href="https://www.codecogs.com/eqnedit.php?latex=\theta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta" title="\theta" /></a> , is used instead. 

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=Q_\theta(S_t,A_t)&space;\leftarrow&space;Q_\theta(S_t,A_t)&plus;\alpha&space;(R_{t&plus;1}&plus;\gamma&space;\,&space;\underset{a'\in&space;A}{max}\:&space;Q_{\theta_t}(S_{t&plus;1},a')-Q_\theta(S_t,A_t))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Q_\theta(S_t,A_t)&space;\leftarrow&space;Q_\theta(S_t,A_t)&plus;\alpha&space;(R_{t&plus;1}&plus;\gamma&space;\,&space;\underset{a'\in&space;A}{max}\:&space;Q_{\theta_t}(S_{t&plus;1},a')-Q_\theta(S_t,A_t))" title="Q_\theta(S_t,A_t) \leftarrow Q_\theta(S_t,A_t)+\alpha (R_{t+1}+\gamma \, \underset{a'\in A}{max}\: Q_{\theta_t}(S_{t+1},a')-Q_\theta(S_t,A_t))" /></a>
</p>

## DDQN
The idea of Double Q-learning is to reduce overestimations by decomposing the max operation in the target into action selection and action evaluation.

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=Q_1(S_t,A_t)&space;\leftarrow&space;Q_1(S_t,A_t)&plus;\alpha&space;(R_{t&plus;1}&plus;\gamma&space;\,&space;Q_2(S_{t&plus;1},\hat{A}_{t&plus;1})-Q_1(S_t,A_t))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Q_1(S_t,A_t)&space;\leftarrow&space;Q_1(S_t,A_t)&plus;\alpha&space;(R_{t&plus;1}&plus;\gamma&space;\,&space;Q_2(S_{t&plus;1},\hat{A}_{t&plus;1})-Q_1(S_t,A_t))" title="Q_1(S_t,A_t) \leftarrow Q_1(S_t,A_t)+\alpha (R_{t+1}+\gamma \, Q_2(S_{t+1},\hat{A}_{t+1})-Q_1(S_t,A_t))" /></a>
</p>
where
<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{A}_{t&plus;1}=\underset{a_{t&plus;1}\in&space;A(S_{t&plus;1})}{argmax}\:&space;Q_1(S_{t&plus;1},A_{t&plus;1})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{A}_{t&plus;1}=\underset{a_{t&plus;1}\in&space;A(S_{t&plus;1})}{argmax}\:&space;Q_1(S_{t&plus;1},A_{t&plus;1})" title="\hat{A}_{t+1}=\underset{a_{t+1}\in A(S_{t+1})}{argmax}\: Q_1(S_{t+1},A_{t+1})" /></a>
</p>

## GORILA
GOogle ReInforcement Learning Architecture (GORILA) is parallelized DQN architecture. 

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

<p align="center">
  <img src="/assets/reinforcement_learning_algorithm_map.png"/>
  <br>
  <b> Fig.1: </b> Reinforcement learning algorithm map
</p>

# Value Based
The value iteration algorithm tries to find an optimal policy by finding the optimal value function.

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

Generally, the following error is used as the evaluation function.

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=J(\theta)&space;=&space;\left\{&space;\begin{matrix}&space;\frac{1}{2}&space;(R_{t&plus;1}&plus;\gamma&space;\,&space;\underset{a'\in&space;A}{max}\:&space;Q_\theta(S_{t&plus;1},a')-Q_\theta(S_t,A_t))^2&space;\;&space;\;&space;\;&space;\;&space;\;&space;|\delta&space;|\leq&space;1&space;\\&space;|R_{t&plus;1}&plus;\gamma&space;\,&space;\underset{a'\in&space;A}{max}\:&space;Q_\theta(S_{t&plus;1},a')-Q_\theta(S_t,A_t)|\;&space;\;&space;\;&space;\;&space;\;\;\;\;\;&space;|\delta&space;|>&space;1&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?J(\theta)&space;=&space;\left\{&space;\begin{matrix}&space;\frac{1}{2}&space;(R_{t&plus;1}&plus;\gamma&space;\,&space;\underset{a'\in&space;A}{max}\:&space;Q_\theta(S_{t&plus;1},a')-Q_\theta(S_t,A_t))^2&space;\;&space;\;&space;\;&space;\;&space;\;&space;|\delta&space;|\leq&space;1&space;\\&space;|R_{t&plus;1}&plus;\gamma&space;\,&space;\underset{a'\in&space;A}{max}\:&space;Q_\theta(S_{t&plus;1},a')-Q_\theta(S_t,A_t)|\;&space;\;&space;\;&space;\;&space;\;\;\;\;\;&space;|\delta&space;|>&space;1&space;\end{matrix}\right." title="J(\theta) = \left\{ \begin{matrix} \frac{1}{2} (R_{t+1}+\gamma \, \underset{a'\in A}{max}\: Q_\theta(S_{t+1},a')-Q_\theta(S_t,A_t))^2 \; \; \; \; \; |\delta |\leq 1 \\ |R_{t+1}+\gamma \, \underset{a'\in A}{max}\: Q_\theta(S_{t+1},a')-Q_\theta(S_t,A_t)|\; \; \; \; \;\;\;\;\; |\delta |> 1 \end{matrix}\right." /></a>
</p>

### Neural fitted Q
It is known that it will degrade the learning stability if the target Q value is calculated with the same parameters <a href="https://www.codecogs.com/eqnedit.php?latex=\theta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta" title="\theta" /></a> as the updating parameters. In order to stablize the learning process, the parameter <a href="https://www.codecogs.com/eqnedit.php?latex=\theta^-" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta^-" title="\theta^-" /></a>, which is a periodic copy of the <a href="https://www.codecogs.com/eqnedit.php?latex=\theta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta" title="\theta" /></a> , is used instead. 

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=Q_\theta(S_t,A_t)&space;\leftarrow&space;Q_\theta(S_t,A_t)&plus;\alpha&space;(R_{t&plus;1}&plus;\gamma&space;\,&space;\underset{a'\in&space;A}{max}\:&space;Q_{\theta_t}(S_{t&plus;1},a')-Q_\theta(S_t,A_t))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Q_\theta(S_t,A_t)&space;\leftarrow&space;Q_\theta(S_t,A_t)&plus;\alpha&space;(R_{t&plus;1}&plus;\gamma&space;\,&space;\underset{a'\in&space;A}{max}\:&space;Q_{\theta_t}(S_{t&plus;1},a')-Q_\theta(S_t,A_t))" title="Q_\theta(S_t,A_t) \leftarrow Q_\theta(S_t,A_t)+\alpha (R_{t+1}+\gamma \, \underset{a'\in A}{max}\: Q_{\theta_t}(S_{t+1},a')-Q_\theta(S_t,A_t))" /></a>
</p>

### Experience replay
When the series of inputs which have strong correlation are used to train a network, the parameters of the network will be updated according to the recent similar inputs, resulting in degrading an estimation for older inputs and preventing from a convergennce of learning. In order to restrain the sampling bias for the training, experience replay was introduced. In experience replay, the obtained experience <a href="https://www.codecogs.com/eqnedit.php?latex=\mathfrak{E}[S_t,A_t,R_{t&plus;1},S_{t&plus;1}]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathfrak{E}[S_t,A_t,R_{t&plus;1},S_{t&plus;1}]" title="\mathfrak{E}[S_t,A_t,R_{t+1},S_{t+1}]" /></a> is stored to the memory (or experience buffer) and later sampled according to the uniform distribution.

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=i\sim&space;U(0,N)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?i\sim&space;U(0,N)" title="i\sim U(0,N)" /></a>
</p>

## DDQN
The idea of Double Q-learning is to reduce overestimations by decomposing the max operation in the target into action selection and action evaluation.

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=Q_1(S_t,A_t)&space;\leftarrow&space;Q_1(S_t,A_t)&plus;\alpha&space;(R_{t&plus;1}&plus;\gamma&space;\,&space;Q_2(S_{t&plus;1},\hat{A}_{t&plus;1})-Q_1(S_t,A_t))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Q_1(S_t,A_t)&space;\leftarrow&space;Q_1(S_t,A_t)&plus;\alpha&space;(R_{t&plus;1}&plus;\gamma&space;\,&space;Q_2(S_{t&plus;1},\hat{A}_{t&plus;1})-Q_1(S_t,A_t))" title="Q_1(S_t,A_t) \leftarrow Q_1(S_t,A_t)+\alpha (R_{t+1}+\gamma \, Q_2(S_{t+1},\hat{A}_{t+1})-Q_1(S_t,A_t))" /></a>
</p>
where
<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{A}_{t&plus;1}=\underset{a'\in&space;A(S_{t&plus;1})}{argmax}\:&space;Q_1(S_{t&plus;1},a')" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{A}_{t&plus;1}=\underset{a'\in&space;A(S_{t&plus;1})}{argmax}\:&space;Q_1(S_{t&plus;1},a')" title="\hat{A}_{t+1}=\underset{a'\in A(S_{t+1})}{argmax}\: Q_1(S_{t+1},a')" /></a>
</p>

## GORILA
GOogle ReInforcement Learning Architecture (GORILA) is parallelized DQN architecture. 

## [Dueling Network](/examples/reinforcement_learning/mountain_car)
The information within a Q function can be divided into two: a part determined mostly by state; and a part influenced by an action choosed. Dueling network separates the Q function into Value, a part that is determined by state, and Advantage, a part that is influenced by the action. This enables the model to learn the parameters that is related to Value every step regardless of action choosed, i.e. the model can learn faster than DQN.

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=Q_\theta(S_t,A_t;\alpha&space;,\beta&space;)=V_\theta(S_t;\beta)&plus;A_\theta(S_t,A_t;\alpha)-\frac{1}{|A|}\sum_{a\in&space;A_t}{}A_\theta(S_t,A_t;\alpha)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Q_\theta(S_t,A_t;\alpha&space;,\beta&space;)=V_\theta(S_t;\beta)&plus;A_\theta(S_t,A_t;\alpha)-\frac{1}{|A|}\sum_{a\in&space;A_t}{}A_\theta(S_t,A_t;\alpha)" title="Q_\theta(S_t,A_t;\alpha ,\beta )=V_\theta(S_t;\beta)+A_\theta(S_t,A_t;\alpha)-\frac{1}{|A|}\sum_{a\in A_t}{}A_\theta(S_t,A_t;\alpha)" /></a>
</p>

<p align="center">
  <img src="/assets/dueling_Q_struct.png"/>
  <br>
  <b> Fig.2: </b> A popular single streamQ-network (top) and the duel-ingQ-network (bottom).
</p>

## Rainbow
Rainbow commbines DQN with six extensions (the number of colors in a rainbow!) that address the limitaions of the original DQN algorithm. The extensions are: 1.DQN, 2.DDQN, 3.Prioritized experience replay, 4.Dueling network, 5.Multi-step learning, 6.Distributional RL, and 7.Noisy network.

### Prioritized experience replay
Experience replay enabled the sampling independent from markov property. However, the sampling was done based on the uniform distribution, and importance of each sample was neglected. Prioritized experience replay resolves the problem by weighting each sample; <a href="https://www.codecogs.com/eqnedit.php?latex=i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?i" title="i" /></a>th sample has importance <a href="https://www.codecogs.com/eqnedit.php?latex=p_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p_i" title="p_i" /></a> of <a href="https://www.codecogs.com/eqnedit.php?latex=p_i=|\delta_i|&plus;\epsilon" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p_i=|\delta_i|&plus;\epsilon" title="p_i=|\delta_i|+\epsilon" /></a>. The sampling is done according to the importance as follows:   

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=i\sim&space;P(i)=\frac{p_i^\alpha}{\sum&space;p_k^\alpha}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?i\sim&space;P(i)=\frac{p_i^\alpha}{\sum&space;p_k^\alpha}" title="i\sim P(i)=\frac{p_i^\alpha}{\sum p_k^\alpha}" /></a>
</p>

The estimation of the expected value with stochastic updates relies on those updates correspondingto the same distribution as its expectation. Prioritized experience replay introduces bias because it changes this distribution in an uncontrolled fashion.

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=w_i&space;=&space;\frac{(N\cdot&space;P(i))^{-\beta}&space;}{\underset{i}{max}\:&space;w_i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w_i&space;=&space;\frac{(N\cdot&space;P(i))^{-\beta}&space;}{\underset{i}{max}\:&space;w_i}" title="w_i = \frac{(N\cdot P(i))^{-\beta} }{\underset{i}{max}\: w_i}" /></a>
</p>

These weights can be folded into the Q-learning update by using <a href="https://www.codecogs.com/eqnedit.php?latex=w_i\delta_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w_i\delta_i" title="w_i\delta_i" /></a> instead of <a href="https://www.codecogs.com/eqnedit.php?latex=\delta_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\delta_i" title="\delta_i" /></a>.

### Multi-step learning
Multi-step learning combines Q-learning and Montecarlo to improve the estimation quality of the value. It takes the future n-steps rewards and the value at n-steps to calculate the <a href="https://www.codecogs.com/eqnedit.php?latex=\delta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\delta" title="\delta" /></a>.     

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\delta&space;=&space;\sum_{i=0}^{n-1}(\gamma^iR_{t&plus;i&plus;1})&plus;\gamma^n\underset{a'\in&space;A}{max}\:&space;Q(S_{t&plus;n},a')-Q(S_t,A_t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\delta&space;=&space;\sum_{i=0}^{n-1}(\gamma^iR_{t&plus;i&plus;1})&plus;\gamma^n\underset{a'\in&space;A}{max}\:&space;Q(S_{t&plus;n},a')-Q(S_t,A_t)" title="\delta = \sum_{i=0}^{n-1}(\gamma^iR_{t+i+1})+\gamma^n\underset{a'\in A}{max}\: Q(S_{t+n},a')-Q(S_t,A_t)" /></a>
</p>

The parameter n is very sensitive but for the Atari games, n=3 is seid to be good.

### Distributional RL
This is also the way to improve the estimation quality of the value. Distributional RL treats the reward as distribution whose mean and variance will reflect the state and action instead of "expectation value," which is basically the average of every rewards.

<p align="center">
  <img src="/assets/distributional_rl.png"/>
  <br>
  <b> Fig.2: </b> Agent differentiates action-value distributions under pressure.
</p>

### Noisy network
Noisy network improves the exploration efficiency by letting the model to learn the exploration rate for itself.

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=y=(W&plus;\sigma^W&space;\odot&space;\epsilon^W)x&plus;(b&plus;\sigma^b&space;\odot&space;\epsilon^b)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y=(W&plus;\sigma^W&space;\odot&space;\epsilon^W)x&plus;(b&plus;\sigma^b&space;\odot&space;\epsilon^b)" title="y=(W+\sigma^W \odot \epsilon^W)x+(b+\sigma^b \odot \epsilon^b)" /></a>
</p>

Note that when <a href="https://www.codecogs.com/eqnedit.php?latex=\sigma=0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sigma=0" title="\sigma=0" /></a>, noisy network is equivalent to the normal linear layer.

# Actor-Critic
## A3C

## A2C
A2C stands for Advantage Actor-Critic. 

## DDPG
Deep Deterministic Policy Gradient (DDPG) is an off-policy, model-free, and actor-critic algorithm. 

## [TD3](/examples/reinforcement_learning/bipedal_walker)
Twin Delayed DDPG (TD3). DDPG is frequently brittle with respect to hyperparameters and tunings. A common failure mode for DDPG is that the learned Q-function begins to dramatically overestimate Q-values, which then leads to the policy breaking, exploiting the errors in the Q-function.

# Policy Based
The policy iteration algorithm manipulates the policy directly, rather than finding it indirectly via the optimal value function.

## TRPO
Trust Region Policy Optimization (TRPO)

## PPO
Proximal Policy Optimization (PPO) is a policy gradient based method. 

## ACKTR
Actor-Critic using Kronecker-Factored Trust Region (ACKTR) is a policy gradient method with the trust region optimization, which will reduce the complexity closer to a first-order optimization like the Gradient Descent. 

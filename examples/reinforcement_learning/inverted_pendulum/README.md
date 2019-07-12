# [CartPole v0](https://github.com/openai/gym/wiki/CartPole-v0)
A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.

<p align="center">
  <img src="/assets/cartpole_random.gif">
</p>

## Details
The agent only knows his `cart position`, `cart velocity`, `pole angle`, and `pole velocity` in evry step. The agent can take one action from `push left` and `push right`.


## Solving with DQN
Following are the commands used to train and test the model:

To train the model:
```bash
python dqn.py train --itr 200 --capacity 10000 --batch 80 --save True --plot True
```

To run with pre-trained weights:
```bash
python dqn.py test
```

## Results
Reward Plot:
<p align="center">
  <img src="/assets/cartpole_loss.png">
</p>

The obtained result:
<p align="center">
  <img src="/assets/cartpole_dqn.gif">
</p>

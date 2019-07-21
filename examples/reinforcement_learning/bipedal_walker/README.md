# [BipedalWalker-v2](https://gym.openai.com/envs/BipedalWalker-v2/)

Get a 2D biped walker to walk through rough terrain by applying motor torque.

## Solving with TD3
Following are the commands used to train and test the model:

To train the model:
```bash
python td3.py train
```

To run with pre-trained weights:
```bash
python td3.py test
```

## Results
The obtained result:
<p align="center">
  <img src="/assets/bipedal_walker_td3.gif">
</p>

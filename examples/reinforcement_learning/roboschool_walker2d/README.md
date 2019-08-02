# [RoboschoolWalker2d-v1](https://gym.openai.com/envs/RoboschoolWalker2d-v1/)

2-D two-legged walking robot similar to MuJoCo Walker2D. The task is to make robot run as fast as possible

<p align="center">
  <img src="/assets/roboschool_walker2d_random.gif">
</p>

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
  <img src="/assets/roboschool_walker2d_td3.gif">
</p>


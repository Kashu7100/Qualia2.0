# [BipedalWalker-v2](https://gym.openai.com/envs/BipedalWalker-v2/)

Get a 2D bipedal walker to walk through rough terrain by applying motor torque.

<p align="center">
  <img src="/assets/bipedal_walker_random.gif">
</p>

## Details
Reward is given for moving forward, total 300+ points up to the far end. If the robot falls, it gets -100. Applying motor torque costs a small amount of points, more optimal agent will get better score. State consists of hull angle speed, angular velocity, horizontal speed, vertical speed, position of joints and joints angular speed, legs contact with ground, and 10 lidar rangefinder measurements. There's no coordinates in the state vector.

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

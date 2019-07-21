# [LunarLanderContinuous-v2](https://gym.openai.com/envs/LunarLanderContinuous-v2/)

<p align="center">
  <img src="/assets/lunar_lander_cont_random.gif">
</p>

## Details
Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector. Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points. If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main engine is -0.3 points each frame. Solved is 200 points. Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land on its first attempt. Action is two real values vector from -1 to +1. First controls main engine, -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power. Second value -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off.

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
  <img src="/assets/lunar_lander_cont_td3.gif">
</p>

# [MountainCar-v0](https://github.com/openai/gym/wiki/MountainCar-v0)
The objective is to get an under powered car to the top of a hill (top = 0.5 position)

<p align="center">
  <img src="/assets/mountaincar_random.gif">
</p>

## Details
The agent only knows his `position` and `velocity` in evry step.
The agent can take one action from three: `push left`, `no push`, and `push right`   

## Usage
Following are the commands used to train and test the model:

To train the model:
```bash
python dueling_network.py train --itr 1000 --capacity 10000 --batch 80 --save True --plot True
```

To run with pre-trained weights:
```bash
python dueling_network.py test
```

## Results
Reward Plot:
<p align="center">
  <img src="/assets/mountaincar_loss.png">
</p>

The obtained result:
<p align="center">
  <img src="/assets/mountaincar_duelingnet.gif">
</p>

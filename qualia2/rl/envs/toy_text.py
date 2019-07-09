# -*- coding: utf-8 -*- 
from ..core import Env, Tensor

class FrozenLake(Env):
    ''' FrozenLake\n
    The agent controls the movement of a character in a grid world. 
    Some tiles of the grid are walkable, and others lead to the agent falling into the water. 
    Additionally, the movement direction of the agent is uncertain and only partially depends on the chosen direction. 
    The agent is rewarded for finding a walkable path to a goal tile.

        SFFF       (S: starting point, safe)
        FHFH       (F: frozen surface, safe)
        FFFH       (H: hole, fall to your doom)
        HFFG       (G: goal, where the frisbee is located)
    
    Reference:
        https://gym.openai.com/envs/FrozenLake-v0/
    '''
    def __init__(self):
        super().__init__('FrozenLake-v0')

    def show(self, filename=None):
        frames = []
        self.env.reset()
        for _ in range(self.max_steps):
            self.env.render()
            self.env.step(self.env.action_space.sample())
            frames.append(self.env.render())
        self.env.close()
        if filename is not None:
            self.animate(frames, filename)
    
class FrozenLake8x8(Env):
    ''' FrozenLake8x8\n
    The agent controls the movement of a character in a grid world. 
    Some tiles of the grid are walkable, and others lead to the agent falling into the water. 
    Additionally, the movement direction of the agent is uncertain and only partially depends on the chosen direction. 
    The agent is rewarded for finding a walkable path to a goal tile.

        SFFF       (S: starting point, safe)
        FHFH       (F: frozen surface, safe)
        FFFH       (H: hole, fall to your doom)
        HFFG       (G: goal, where the frisbee is located)
    
    Reference:
        https://gym.openai.com/envs/FrozenLake8x8-v0/
    '''
    def __init__(self):
        super().__init__('FrozenLake8x8-v0')

    def show(self, filename=None):
        frames = []
        self.env.reset()
        for _ in range(self.max_steps):
            self.env.render()
            self.env.step(self.env.action_space.sample())
            frames.append(self.env.render())
        self.env.close()
        if filename is not None:
            self.animate(frames, filename)

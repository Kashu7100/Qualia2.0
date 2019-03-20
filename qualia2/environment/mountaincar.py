# -*- coding: utf-8 -*- 
from ..config import gpu
from ..core import *
from .env import *

class MountainCar(Environment):
    ''' MountainCar\n
    Get an under powered car to the top of a hill (top = 0.5 position)
    
    Observation: 
        Type: Box(2)
        Num	Observation    Min      Max
        0 	position 	  -1.2  	0.6
        1 	velocity      -0.07 	0.07
        
    Actions:
        Type: Discrete(3)
        Num	Action
        0 	push left
        1 	no push
        2 	push right
    
    Reward:
        -1 for each step

    Reference:
        https://github.com/openai/gym/wiki/MountainCar-v0
    '''
    def __init__(self, agent, max_step, max_episodes):
        super().__init__('MountainCar-v0', agent, max_step, max_episodes)
    
    def run(self):
        self.frames = []
        for episode in range(self.max_episodes):
            state = self.env.reset()
            for step in range(self.max_steps):
                action = self.agent(state, episode, 3)
                nextstate, reward, done, _ = self.env.step(int(action[0]))
                if done:
                    nextstate = np.zeros((self.num_states))
                self.agent.memorize(state, action, nextstate, reward)    
                self.agent.experience_replay()
                state = nextstate
                if episode == self.max_episodes-1:
                    self.frames.append(self.env.render(mode='rgb_array'))
                if done:
                    print('[*] episode {}: finished after {} steps'.format(episode+1, step+1))
                    if self.agent.target_model is not None:
                        if(episode%2==0):
                            self.agent.update_target_model()
                    break

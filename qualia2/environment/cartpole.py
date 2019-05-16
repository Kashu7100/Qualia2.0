# -*- coding: utf-8 -*- 
from ..core import *
from .env import *

class CartPole(Environment):
    ''' CartPole\n
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. 
    The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.
    
    Observation: 
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8          4.8
        1	Cart Velocity             -Inf          Inf
        2	Pole Angle                -24 deg       24 deg
        3	Pole Velocity At Tip      -Inf          Inf
        
    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right

    Reward:
        0 for each step
        -1 if terminate condition meet before max_steps-10
        1 if terminate condition meet after max_steps-10
        (Note: original reward with the gym environment is not used)
    
    Reference:
        https://github.com/openai/gym/wiki/CartPole-v0
    '''
    def __init__(self, agent, max_step, max_episodes):
        super().__init__('CartPole-v0', agent, max_step, max_episodes)

    def run(self):
        self.frames = []
        for episode in range(self.max_episodes):
            state = self.env.reset()
            for step in range(self.max_steps):
                action = self.agent(state, episode, 2)
                nextstate, _, done, _ = self.env.step(int(action[0]))
                if done:
                    nextstate = np.zeros((self.num_states))
                    if step < self.max_steps-10:
                        reward = -1.0
                    else:
                        reward = 1.0
                else:
                    reward = 0.0
                self.agent.memorize(state, action, nextstate, reward)    
                self.agent.experience_replay()
                state = nextstate
                if episode == self.max_episodes-1:
                    self.frames.append(self.env.render(mode='rgb_array'))
                if done:
                    if len(self.rewards) > 0:
                        if max(self.rewards) < step+1 or episode == self.max_episodes-1:
                            self.agent.save(self.path+'/tmp/{}-ep{}-sc{}'.format(self.__class__.__name__, episode+1, step+1))
                    self.rewards.append(step+1)
                    logger.debug('[*] episode {}: finished after {} steps'.format(episode+1, step+1))
                    if self.agent.target_model is not None:
                        if(episode%2==0):
                            self.agent.update_target_model()
                    break
        logger.info('[*] training finished. best score: {}'.format(max(self.rewards)))

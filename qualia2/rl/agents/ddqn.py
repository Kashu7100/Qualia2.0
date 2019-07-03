# -*- coding: utf-8 -*- 
from ..core import Agent, max, np
from ..util import Trainer

class DDQN(Agent):
    '''DQN 2015 implementation\n
    This implementation uses double networks for learning. 
    DQN class incopolates the model (Module) and the optim (Optimizer).
    The model learns with experience replay, which is implemented in update() method.
    '''
    def __init__(self, eps, actions):
        super().__init__(eps, actions)

    def train_signal(self, experience, gamma):
        self.model.eval()
        self.target.eval()
        state, next_state, reward, action, done = experience
        # get state action value
        action_value = self.model(state).gather(1, action) 
        action_next = max(self.target(next_state), axis=1)
        action_next[np.all(next_state.data==0, axis=1)] = 0
        target_action_value = reward + gamma*action_next
        return action_value, target_action_value.detach()

class DDQNTrainer(Trainer):
    def __init__(self, memory, batch=64, capacity=2048, gamma=0.9):
        super().__init__(memory, batch, capacity, gamma)    

    def after_episode(self, episode, steps, agent, loss, reward, filename=None):
        super().after_episode(episode, steps, agent, loss, reward, filename)
        agent.target.load_state_dict(agent.model.state_dict())

    def train(self, env, model, optim, episodes=200, render=False, filename=None):
        agent = DDQN.reload(env, model)
        agent.set_optim(optim)
        self.before_train(env, agent)
        self.train_routine(env, agent, episodes=episodes, render=render, filename=filename)
        return agent
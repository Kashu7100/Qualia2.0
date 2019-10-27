# -*- coding: utf-8 -*- 
from ..rl_core import ValueAgent, np
from ..rl_util import Trainer

class DDQN(ValueAgent):
    '''DQN 2015 implementation\n
    This implementation uses double networks for learning. 
    DQN class incopolates the model (Module) and the optim (Optimizer).
    The model learns with experience replay, which is implemented in update() method.
    '''
    def __init__(self, eps, actions):
        super().__init__(eps, actions)

    def get_train_signal(self, experience, gamma):
        self.model.eval()
        self.target.eval()
        state, next_state, reward, action, done = experience
        state_action_value = self.model(state).gather(1, action) 
        action_next = np.argmax(self.model(next_state).data, axis=1).reshape(-1,1)
        next_state_action_value = self.target(next_state).gather(1, action_next) 
        next_state_action_value[done] = 0
        target_action_value = reward + gamma * next_state_action_value
        return state_action_value, target_action_value.detach()

class DDQNTrainer(Trainer):
    '''DDQNTrainer \n
    Args:
        memory (deque): replay memory object
        capacity (int): capacity of the memory
        batch (int): batch size for training
        gamma (int): gamma value
        target_update_interval (int): interval for updating target network
    '''
    def __init__(self, memory, batch, capacity, gamma=0.99, target_update_interval=3):
        super().__init__(memory, batch, capacity, gamma)   
        self.target_update_interval = target_update_interval 

    def after_episode(self, episode, steps, agent, loss, reward, filename=None):
        super().after_episode(episode, steps, agent, loss, reward, filename)
        if(episode%self.target_update_interval==0):
            agent.update_target_model()

    def train(self, env, agent, episodes=200, render=True, filename=None):
        self.before_train(env, agent)
        self.train_routine(env, agent, episodes=episodes, render=render, filename=filename)
        self.after_train()
        return agent

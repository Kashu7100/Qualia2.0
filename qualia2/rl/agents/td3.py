# -*- coding: utf-8 -*- 
from ..core import ActorCriticAgent, np, Tensor
from ...functions import minimum, mse_loss, mean
from ..util import Trainer, Experience
import numpy

class TD3(ActorCriticAgent):
    ''' Twin Delayed DDPG (TD3) \n
    Args:
        actor (Module): actor network
        critic (Module): critic network
    '''
    def __init__(self, actor, critic):
        super().__init__(actor, critic)
        self.critic2 = critic
        self.critic2_target = critic
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_optim = None

    def set_critic_optim(self, optim, **kwargs):
        self.critic_optim = optim(self.critic.params, **kwargs)
        self.critic2_optim = optim(self.critic2.params, **kwargs)
    
    def save(self, filename):
        self.actor.save(filename+'_actor')
        self.critic.save(filename+'_critic1')
        self.critic2.save(filename+'_critic2')

    def load(self, filename):
        self.actor.load(filename+'_actor')
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic.load(filename+'_critic1')
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic2.load(filename+'_critic2')
        self.critic2_target.load_state_dict(self.critic2.state_dict())

    def get_train_signal(self, experience, max_action, policy_noise, noise_clip, gamma=0.9):
        state, next_state, reward, action, done = experience
        # Select next action according to target policy:
        noise = Tensor(np.random.normal(0, policy_noise, size=action.shape))
        noise = noise.clamp(-noise_clip, noise_clip)
        next_action = self.actor_target(next_state) + noise
        next_action = next_action.clamp(-max_action, max_action)

        # Compute target Q-value:
        target_Q1 = self.critic_target(next_state, next_action)
        target_Q2 = self.critic2_target(next_state, next_action)
        target_Q = minimum(target_Q1, target_Q2)
        target_Q = (reward + (1-Tensor(done)) * gamma * target_Q).detach()
        return state, Tensor(action), target_Q

    def update_critic1(self, state, action, target):
        current_Q1 = self.critic(state, action)
        loss_Q1 = mse_loss(current_Q1, target)
        self.critic_optim.zero_grad()
        loss_Q1.backward()
        self.critic_optim.step()

    def update_critic2(self, state, action, target):
        current_Q2 = self.critic2(state, action)
        loss_Q2 = mse_loss(current_Q2, target)
        self.critic2_optim.zero_grad()
        loss_Q2.backward()
        self.critic2_optim.step()
    
    def update_actor(self, state):
        actor_loss = -mean(self.critic(state, self.actor(state)),axis=0)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        return float(actor_loss.asnumpy()[0])
    
class TD3Trainer(Trainer):
    ''' TD3 Trainer \n
    Args:
        memory (deque): replay memory object
        capacity (int): capacity of the memory
        batch (int): batch size for training
        gamma (int): gamma value
        policy_delay (int): interval for updating target network
    '''
    def __init__(self, memory, batch, capacity, gamma=0.99, polyak=0.995, policy_delay=2, exploration_noise=0.1, policy_noise=0.2, noise_clip=0.5):
        super().__init__(memory, batch, capacity, gamma)   
        self.polyak = polyak
        self.exploration_noise = exploration_noise
        self.policy_delay = policy_delay 
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip

    def before_episode(self, env, agent):
        self.max_action = float(env.action_space.high[0])
        return env.reset(), False, 0

    def train_routine(self, env, agent, episodes=200, render=False, filename=None):
        try:
            for episode in range(episodes):
                state, done, steps = self.before_episode(env, agent)
                tmp_loss = []
                tmp_reward = []
                while not done:
                    if render and (episode+1)%10==0:
                        env.render()
                    action = agent(state)
                    action = action + numpy.random.normal(0, self.exploration_noise, size=env.action_space.shape[0])
                    action = action.clip(env.action_space.low, env.action_space.high)

                    next, reward, done, _ = env.step(action)
                    self.memory.append(Experience(state, next, reward, action, done))
                    state = next
                    steps += 1
                    if done or steps == env.max_steps-1:
                        if len(self.memory) > self.batch:
                            tmp_loss.append(self.experience_replay(episode, steps, agent))
                    tmp_reward.append(reward.data[0])                
                    
                if render and (episode+1)%10==0:
                    env.close()
                self.after_episode(episode+1, steps, agent, tmp_loss, tmp_reward, filename)
        except:
            import os
            path = os.path.dirname(os.path.abspath(__file__))
            if not os.path.exists(path + '/tmp/'):
                os.makedirs(path + '/tmp/') 
            agent.save(path + '/tmp/auto_save')
            raise Exception('[*] Training aborted.')

    def experience_replay(self, episode, step_count, agent):
        loss = 0
        for i in range(step_count):
            experience, idx, weights = self.memory.sample(self.batch)
            state, action, target_Q = agent.get_train_signal(experience, self.max_action, self.policy_noise, self.noise_clip, self.gamma)
            agent.update_critic1(state, action, target_Q)
            agent.update_critic2(state, action, target_Q)
            # delayed policy updates
            if i%self.policy_delay == 0:
                loss += agent.update_actor(state)

                # polyak averaging update:
                for param, target_param in zip(agent.actor.params(), agent.actor_target.params()):
                    target_param.copy((self.polyak * target_param.data) + ((1-self.polyak) * param.data))
                
                for param, target_param in zip(agent.critic.params(), agent.critic_target.params()):
                    target_param.copy((self.polyak * target_param.data) + ((1-self.polyak) * param.data))
                
                for param, target_param in zip(agent.critic2.params(), agent.critic2_target.params()):
                    target_param.copy((self.polyak * target_param.data) + ((1-self.polyak) * param.data))
        return loss

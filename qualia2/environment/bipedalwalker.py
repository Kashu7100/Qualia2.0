# -*- coding: utf-8 -*- 
#from ..config import gpu
from ..core import *
from .env import *

class BipedalWalker(Environment):
    '''BipedalWalker \n
    Get a 2D biped walker to walk through rough terrain.
    Observation:
        Type: Box(24)
        Num 	Observation 	        Min 	Max 	Mean
        0 	    hull_angle 	            0 	    2*pi 	0.5
        1 	    hull_angularVelocity 	-inf 	+inf 	-
        2 	    vel_x 	                -1 	    +1 	-
        3 	    vel_y 	                -1 	    +1 	-
        4   	hip_joint_1_angle 	    -inf 	+inf 	-
        5   	hip_joint_1_speed 	    -inf 	+inf 	-
        6   	knee_joint_1_angle 	    -inf 	+inf 	-
        7   	knee_joint_1_speed 	    -inf 	+inf 	-
        8   	leg_1_ground_contact_flag 	0 	1 	-
        9   	hip_joint_2_angle 	    -inf 	+inf 	-
        10  	hip_joint_2_speed 	    -inf 	+inf 	-
        11  	knee_joint_2_angle 	    -inf 	+inf 	-
        12     	knee_joint_2_speed 	    -inf 	+inf 	-
        13 	    leg_2_ground_contact_flag 	0 	1 	-
        14-23 	10 lidar readings 	    -inf 	+inf 	-    
    Actions:
        Type: Box(4) - Torque control(default)
        Num 	Name     	                Min 	Max
        0 	    Hip_1 (Torque / Velocity) 	-1 	+1
        1 	    Knee_1 (Torque / Velocity) 	-1 	+1
        2 	    Hip_2 (Torque / Velocity) 	-1 	+1
        3 	    Knee_2 (Torque / Velocity) 	-1 	+1
    Rewards:
        Reward is given for moving forward, total 300+ points up to the far end. If the robot falls, it gets -100. 
        Applying motor torque costs a small amount of points, more optimal agent will get better score. 
    Reference:
        https://github.com/openai/gym/wiki/BipedalWalker-v2
    '''
    def __init__(self, agent, max_step=2000, max_episodes=1000):
        super().__init__('BipedalWalker-v2', agent, max_step, max_episodes)

    def run(self):
        self.frames = []
        for episode in range(self.max_episodes):
            state = self.env.reset()
            for step in range(self.max_steps):
                action = self.agent(state)
                # add exploration noise
                action += np.random.normal(0, exploration_noise, size=self.env.action_space.shape[0])
                action = action.clip(env.action_space.low, env.action_space.high)

                nextstate, reward, done, _ = self.env.step(action)
                if done:
                    nextstate = np.zeros((self.num_states))
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

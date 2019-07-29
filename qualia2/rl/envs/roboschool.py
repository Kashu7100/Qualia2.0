# -*- coding: utf-8 -*- 
from ..core import Env, Tensor, np
import roboschool

class RoboSchoolBase(Env):
    def __init__(self, env):
        super().__init__(env)

    def show(self, filename=None):
        frames = []
        try:
            #self.env.render()
            self.env.reset()
            while True:
                _, _, done, _ = self.env.step(self.env.action_space.sample())
                if filename is not None:
                    frames.append(self.env.render(mode='rgb_array'))
                else:
                    self.env.render()
                if done:
                    break
            self.env.close()
            if filename is not None:
                self.animate(frames, filename)
        except:
            self.env.close()
            raise Exception('[*] Exception occurred during the Env.show() process.')

class RoboschoolAnt(RoboSchoolBase):
    def __init__(self):
        super().__init__('RoboschoolAnt-v1')

#class RoboschoolAtlasForwardWalk(RoboSchoolBase):
#    def __init__(self):
#        super().__init__('RoboschoolAtlasForwardWalk-v1')

class RoboschoolHalfCheetah(RoboSchoolBase):
    def __init__(self):
        super().__init__('RoboschoolHalfCheetah-v1')

class RoboschoolHumanoid(RoboSchoolBase):
    def __init__(self):
        super().__init__('RoboschoolHumanoid-v1')

class RoboschoolWalker2d(RoboSchoolBase):
    def __init__(self):
        super().__init__('RoboschoolWalker2d-v1')        

class RoboschoolHopper(RoboSchoolBase):
    def __init__(self):
        super().__init__('RoboschoolHopper-v1')     
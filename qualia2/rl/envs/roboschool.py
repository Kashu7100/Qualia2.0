# -*- coding: utf-8 -*- 
from ..core import Env, Tensor, np
try:
    import roboschool
    
    class RoboSchoolBase(Env):
        ''' RoboSchoolBase \n
        '''
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
        ''' RoboschoolAnt \n
        Observation:
            Type: Box(28,)
        
        Actions:
            Type: Box(8,)
        '''
        def __init__(self):
            super().__init__('RoboschoolAnt-v1')

    class RoboschoolHalfCheetah(RoboSchoolBase):
        ''' RoboschoolHalfCheetah \n
        Observation:
            Type: Box(26,)
        
        Actions:
            Type: Box(6,)    
        '''
        def __init__(self):
            super().__init__('RoboschoolHalfCheetah-v1')

    class RoboschoolHumanoid(RoboSchoolBase):
        ''' RoboschoolHumanoid \n
        Observation:
            Type: Box(44,)
        
        Actions:
            Type: Box(17,)    
        '''
        def __init__(self):
            super().__init__('RoboschoolHumanoid-v1')

    class RoboschoolWalker2d(RoboSchoolBase):
        ''' RoboschoolHumanoid \n
        Observation:
            Type: Box(22,)
        
        Actions:
            Type: Box(6,)    
        '''
        def __init__(self):
            super().__init__('RoboschoolWalker2d-v1')        

    class RoboschoolHopper(RoboSchoolBase):
        ''' RoboschoolHumanoid \n
        Observation:
            Type: Box(15,)
        
        Actions:
            Type: Box(3,)    
        '''
        def __init__(self):
            super().__init__('RoboschoolHopper-v1')

except:
    print('[*] install roboschool to use roboschool environment.')
    print('[*] pip install roboschool')
# -*- coding: utf-8 -*- 
from ... import to_cpu
from ...core import *
from ...autograd import Tensor
from collections import OrderedDict 
import h5py as h5 
from logging import getLogger
logger = getLogger('QualiaLogger').getChild('module')

class Module(object):
    '''Base class for all neural network modules in qualia.\n 
    Module can incoporate Modules, allowing to nest them in a tree structure.  
    ''' 
    def __init__(self): 
        self._modules = OrderedDict() 
        self._params = OrderedDict() 
        self.training = True 
        self.num_params = 0
        self.input_shape = None
        self.output_shape = None
    
    def __repr__(self):
        print('{}('.format(self.__class__.__name__))
        if self._modules: 
            for i, (name, module) in enumerate(self._modules.items()):
                print('    [{}] {}: {}'.format(i, name, repr(module)))
        return ') at 0x{:0{}X}'.format(id(self), 16)

    def summary(self, input_shape, *args):
        logger.info('-'*76)
        logger.info('{:^76}'.format('Model: ' + self.__class__.__name__))
        if type(input_shape) is list:
            raise NotImplementedError

        elif type(input_shape) is tuple:
            x = Tensor(np.zeros(input_shape), requires_grad=False)
            if self._modules: 
                logger.info('{}\n| {:20}|{:^20}|{:^20}|{:^10}|\n{}'.format('-'*76, 'layers', 'input shape', 'output shape', 'params #', '='*76))
                for _, module in self._modules.items():
                    module.input_shape = None
                    module.output_shape = None
                total_params = 0
                _ = self.forward(x, *args)
                for i, (_, module) in enumerate(self._modules.items()):
                    logger.info('| {:20}|{:^20}|{:^20}|{:^10}|'.format(module.__class__.__name__+'-'+str(i), str(module.input_shape), str(module.output_shape), str(module.num_params)))
                    total_params += module.num_params
        logger.info('='*76)
        logger.info('total params: {}'.format(total_params))
        logger.info('training mode: {}'.format(self.training))
        logger.info('-'*76)
    
    def __setattr__(self, key, value): 
        if isinstance(value, Module):
            self._modules[key] = value 
        elif isinstance(value, Tensor):
            self._params[key] = value 
        elif type(value) is list:
            if all(isinstance(n, Tensor) for n in value):
                self._params[key] = value
        else:
            object.__setattr__(self, key, value) 
     
    def __getattr__(self, key):
        if self._modules:
            return self._modules[key]
        elif self._params:
            return self._params[key]
     
    def __call__(self, *args, **kwargs): 
        return self.forward(*args, **kwargs) 
     
    def forward(self, *args, **kwargs): 
        raise NotImplementedError 

    def apply(self, fn):
        if not self._modules:
            fn(self)
        else:
            for _, module in self._modules.items():
                fn(module)

    def modules(self):
        if not self._modules: 
            raise Exception('No module found inside {} at 0x{:0{}X}'.format(self.__class__.__name__, id(self), 16))
        else:
            for _, module in self._modules.items(): 
                yield module 

    def params(self): 
        if not self._modules: 
            for _, var in self._params.items(): 
                if type(var) is list:
                    for i in var:
                        yield i
                else:
                    yield var 
        else:
            for _, module in self._modules.items(): 
                for _, var in module._params.items(): 
                    if type(var) is list:
                        for i in var:
                            yield i
                    else:
                        yield var 
     
    def zero_grad(self): 
        if not self._modules: 
            for _, var in self._params.items(): 
                if type(var) is list:
                    for i in var:
                        i.grad = None 
                else:
                    var.grad = None
        else: 
            for _, module in self._modules.items(): 
                for _, var in module._params.items(): 
                    if type(var) is list:
                        for i in var:
                            i.grad = None 
                    else:
                        var.grad = None
    
    def eval(self):
        self.training = False
        if self._modules:
            for _, module in self._modules.items():
                module.training = False
    
    def train(self):
        self.training = True
        if self._modules:
            for _, module in self._modules.items():
                module.training = True

    def state_dict(self):
        '''Returns a dictionary containing a whole state of the module.\n
        '''
        if not self._modules:
            return self._params
        else: 
            return self._modules
    
    def load_state_dict(self, state_dict):
        '''Copies parameters from state_dict into this module.\n
        '''
        if not self._modules: 
            for key, value in state_dict.items(): 
                if type(value) is list:
                    for i, val in enumerate(value):
                        self._params[key][int(i)].data = val.data
                else:
                    self._params[key].data = value.data
        else: 
            for name, module in state_dict.items(): 
                for key, value in module._params.items(): 
                    if type(value) is list:
                        for i, val in enumerate(value):
                            self._modules[name]._params[key][int(i)].data = val.data
                    else:
                        self._modules[name]._params[key].data = value.data

    def save(self, filename, dtype='float64'): 
        '''Saves internal parameters of the Module in HDF5 format.\n 
        Args: 
            filename (str): specify the filename as well as the saving path without the file extension. (ex) path/to/filename 
        ''' 
        with h5.File(filename + '.hdf5', 'w') as file: 
            if not self._modules:
                for key, value in self._params.items():
                    if type(value) is list:
                        grp = file.create_group(str(key)) 
                        for i, val in enumerate(value):
                            if gpu:
                                grp.create_dataset(str(i), dtype=dtype, data=to_cpu(val.data)) 
                            else:
                                grp.create_dataset(str(i), dtype=dtype, data=val.data) 
                    else:
                        if gpu:
                            file.create_dataset(str(key), dtype=dtype, data=to_cpu(value.data)) 
                        else:
                            file.create_dataset(str(key), dtype=dtype, data=value.data) 
            else: 
                for name, module in self._modules.items(): 
                    grp = file.create_group(str(name)) 
                    for key, value in module._params.items(): 
                        if type(value) is list:
                            subgrp = grp.create_group(str(key)) 
                            for i, val in enumerate(value):
                                if gpu:
                                    subgrp.create_dataset(str(i), dtype=dtype, data=to_cpu(val.data)) 
                                else:
                                    subgrp.create_dataset(str(i), dtype=dtype, data=val.data) 
                        else:
                            if gpu:
                                grp.create_dataset(str(key), dtype=dtype, data=to_cpu(value.data)) 
                            else:
                                grp.create_dataset(str(key), dtype=dtype, data=value.data) 
     
    def load(self, filename): 
        '''Loads parameters saved in HDF5 format to the Module.\n 
        Args: 
            filename (str): specify the filename as well as the path to the file without the file extension. (ex) path/to/filename 
        ''' 
        with h5.File(filename + '.hdf5', 'r') as file: 
            if not self._modules: 
                for key in file: 
                    if isinstance(file[key], h5.Group):
                        for i in file[key]:
                            self._params[key][int(i)].data = np.array(file[key][i])
                    else:
                        self._params[key].data = np.array(file[key]) 
            else: 
                for module in file: 
                    for key in file[module]: 
                        if isinstance(file[module][key], h5.Group):
                            for i in file[module][key]:
                                self._modules[module]._params[key][int(i)].data = np.array(file[module][key][i])
                        else:
                            self._modules[module]._params[key].data = np.array(file[module][key])

class Sequential(Module): 
    r'''A sequential container.\n 
    Modules will be added to it in the order they are passed in the constructor.  
 
    Examples:: 
        >>> # model can be defiened by adding Modules 
        >>> model = Sequential( 
        >>>     nn.Conv2d(1,20,5), 
        >>>     nn.ReLU(), 
        >>>     nn.Conv2d(20,64,5), 
        >>>     nn.ReLU()
        >>>     ) 
        >>> # name for each layers can also be specified 
        >>> model = Sequential( 
        >>>     'conv1' = nn.Conv2d(1,20,5), 
        >>>     'relu1' = nn.ReLU(), 
        >>>     'conv2' = nn.Conv2d(20,64,5), 
        >>>     'relu2' = nn.ReLU() 
        >>>     ) 
    ''' 
    def __init__(self, *args, **kwargs): 
        super().__init__() 
        for i, module in enumerate(args): 
            if isinstance(module, Module): 
                self._modules[str(i)] = module 
        for name, module in kwargs.items(): 
            if isinstance(module, Module): 
                self._modules[name] = module 
         
    def __call__(self, x): 
        return self.forward(x)

    def forward(self, x):
        for _, module in self._modules.items(): 
            x = module.forward(x) 
        return x 
     
    def append(self, *arg, **kwarg): 
        if len(arg) > 1 or len(kwarg) > 1: 
            raise Exception('Too much arguments were given.') 
        if isinstance(arg, Module): 
            self._modules[str(len(self._modules))] = arg 
        for name, module in kwarg.items(): 
            if isinstance(module, Module): 
                self._modules[name] = module 
            else: 
                raise Exception('Invalid argument was given. Failed to append.')

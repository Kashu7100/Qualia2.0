# -*- coding: utf-8 -*- 
from ...core import *
from ...autograd import Tensor
from collections import OrderedDict 
from itertools import chain, islice
import h5py as h5
import _pickle as pickle
import gzip
from logging import getLogger
logger = getLogger('QualiaLogger').getChild('module')

class Module(object):
    '''Base class for all neural network modules in qualia.\n 
    Module can incoporate Modules, allowing to nest them in a tree structure.  
    Note that a user-defined model must have super().__init__() in the __init__ of the model.

    Examples::
        >>> # define a module
        >>> class Model(nn.Module):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        >>>         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        >>>         self.fc1 = nn.Linear(500, 50)
        >>>         self.fc2 = nn.Linear(50, 10)
        >>>     def forward(self, x):
        >>>         x = F.relu(F.maxpool2d(self.conv1(x), (2,2)))
        >>>         x = F.relu(F.maxpool2d(self.conv2(x), (2,2)))
        >>>         x = F.reshape(x,(-1, 500))
        >>>         x = F.relu(self.fc1(x))
        >>>         x = self.fc2(x)
        >>>         return x
    ''' 
    def __init__(self): 
        self._modules = OrderedDict() 
        self._params = OrderedDict() 
        self.training = True 
        self.num_params = 0
        self.input_shape = None
        self.output_shape = None
    
    def __repr__(self):
        result = '{}(\n'.format(self.__class__.__name__)
        if self._modules: 
            for i, (name, module) in enumerate(self._modules.items()):
                result += '    [{}] {}: {}\n'.format(i, name, repr(module))
        if __debug__:
            result += ') at 0x{:0{}X}\n'.format(id(self), 16)
        else:
            result += ')\n'
        return result
    
    def __str__(self):
        return self.__class__.__name__

    def _module_summary(self):
        if not self._modules:
            logger.info('| {:20}|{:^20}|{:^20}|{:^10}|'.format(self.__class__.__name__, str(self.input_shape), str(self.output_shape), str(self.num_params)))
            return self.num_params
        else:
            total_params = 0
            for _, module in self._modules.items():
                total_params += module._module_summary()
            return total_params
        
    def summary(self, input_shape):
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
                self.forward(x)
                total_params = self._module_summary()
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
    
    def add_module(self, name, module):
        assert isinstance(module, Module)
        self._modules[name] = module 
     
    def forward(self, *args, **kwargs): 
        raise NotImplementedError 

    def apply(self, fn):
        if not self._modules:
            fn(self)
        else:
            for _, module in self._modules.items():
                module.apply(fn)

    def modules(self):
        if not self._modules: 
            yield self
        else:
            for _, module in self._modules.items(): 
                for var in module.modules():
                    yield var

    def params(self): 
        if self._modules:
            for _, module in self._modules.items(): 
                for var in module.params():
                    yield var
        for _, var in self._params.items(): 
            if type(var) is list:
                for i in var:
                    yield i
            else:
                yield var 

    def zero_grad(self): 
        if self._modules:
            for _, module in self._modules.items(): 
                module.zero_grad()
        for _, var in self._params.items(): 
            if type(var) is list:
                for i in var:
                    i.grad = None 
            else:
                var.grad = None
    
    def eval(self):
        if self._modules:
            for _, module in self._modules.items():
                module.eval()
        self.training = False
    
    def train(self):
        if self._modules:
            for _, module in self._modules.items():
                module.train()
        self.training = True

    def _create_state_dict(self, name=''):
        state_dict = {}
        if self._modules: 
            for key, module in self._modules.items():
                state_dict.update(module._create_state_dict(name+str(key)+'.'))
        for key, value in self._params.items(): 
            if type(value) is list: 
                for i, val in enumerate(value):
                    state_dict[name+str(key)+'.'+str(i)] = val.data 
            else:
                state_dict[name+str(key)] = value.data
        return state_dict
    
    def state_dict(self):
        '''Returns a dictionary containing a whole state of the module.\n
        '''
        state_dict = self._create_state_dict()
        return state_dict
    
    def load_state_dict(self, state_dict, name=''):
        '''Copies parameters from state_dict into this module.\n
        '''
        if self._modules: 
            for key, module in self._modules.items():
                module.load_state_dict(state_dict, name+str(key)+'.')
        for key, value in self._params.items(): 
            if type(value) is list:
                for i, val in enumerate(value):
                    self._params[key][int(i)].data = np.copy(state_dict[name+str(key)+'.'+str(i)].astype(self._params[key][int(i)].dtype))
            else:
                self._params[key].data = np.copy(state_dict[name+str(key)].astype(self._params[key].dtype))
                
    def load_state_dict_from_url(self, url, version=0):
        '''Downloads and copies parameters from the state_dict at the url into this module.\n
        '''
        if not os.path.exists(home_dir+'/pretrained/'):
            os.makedirs(home_dir+'/pretrained/')
        model_dir = home_dir+'/pretrained'
        from urllib.parse import urlparse
        parts = urlparse(url)
        filename = os.path.basename(parts.path)
        cache = os.path.join(model_dir, filename)
        if not os.path.exists(cache): 
            from urllib.request import urlretrieve
            urlretrieve(url, cache, reporthook=download_progress) 
        self.load(cache, version)
    
    def __save__(self, h5file):
        if self._modules: 
            for name, module in self._modules.items(): 
                grp = h5file.create_group(str(name))
                module.__save__(grp)
        for key, value in self._params.items(): 
            if type(value) is list:
                grp = h5file.create_group(str(key)) 
                for i, val in enumerate(value):
                    grp.create_dataset(str(i), dtype='f8', data=val.asnumpy()) 
            else:
                h5file.create_dataset(str(key), dtype='f8', data=value.asnumpy())          

    def __load__(self, h5file):
        if self._modules: 
            for name, module in self._modules.items():
                module.__load__(h5file[name])
        for key, value in self._params.items(): 
            if type(value) is list:
                for i, val in enumerate(value):
                    self._params[key][int(i)].data = np.array(h5file[key][str(i)])
            else:
                self._params[key].data = np.array(h5file[key])
        
    def save(self, filename, dtype='float32', protocol=-1, version=0):
        '''Saves internal parameters of the Module in HDF5 format.\n 
        Args: 
            filename (str): specify the filename as well as the saving path without the file extension. (ex) path/to/filename.qla 
            dtype (str): data type to save
            protocol (int): pickle protocol
            version (int): version for the way of saving. version 1 takes less disk space. 
        ''' 
        if version == 1:
            with gzip.open(filename, 'wb') as f:
                pickle.dump(self.state_dict(dtype), f, protocol)
        elif version == 0:
            with h5.File(filename, 'w') as file: 
                self.__save__(file)

    def load(self, filename, version=0):
        '''Loads parameters saved in HDF5 format to the Module.\n 
        Args: 
            filename (str): specify the filename as well as the path to the file with the file extension. (ex) path/to/filename.qla 
            version (int): version for the way of saving. 
        ''' 
        if version == 1:
            with gzip.open(filename, 'rb') as f:
                self.load_state_dict(pickle.load(f))
        elif version == 0:
            with h5.File(filename, 'r') as file: 
                self.__load__(file)

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
        >>>     conv1 = nn.Conv2d(1,20,5), 
        >>>     relu1 = nn.ReLU(), 
        >>>     conv2 = nn.Conv2d(20,64,5), 
        >>>     relu2 = nn.ReLU() 
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
    
    def __getitem__(self, slice):
        return Sequential(*list(islice(self._modules.values(), slice.start, slice.stop))) 
         
    def __call__(self, x): 
        return self.forward(x)

    def forward(self, x):
        for _, module in self._modules.items(): 
            x = module.forward(x) 
        return x 
     
    def append(self, *arg, **kwarg): 
        if len(arg) > 1 and len(kwarg) > 1: 
            raise Exception('Too much arguments were given.') 
        for module in arg: 
            if isinstance(module, Module): 
                self._modules[str(len(self._modules))] = module
            else: 
                raise Exception('Invalid argument was given. Failed to append.')
        for name, module in kwarg.items(): 
            if isinstance(module, Module): 
                self._modules[name] = module 
            else: 
                raise Exception('Invalid argument was given. Failed to append.')

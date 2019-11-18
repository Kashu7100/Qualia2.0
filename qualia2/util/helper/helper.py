# -*- coding: utf-8 -*- 
from ...core import *
import time
from datetime import timedelta
from enum import Enum
import matplotlib.pyplot as plt
from logging import getLogger
logger = getLogger('QualiaLogger').getChild('helper')

class Events(Enum):
    STARTED = 0
    COMPLETED = 1
    EPOCH_STARTED = 2
    EPOCH_COMPLETED = 3
    ITR_STARTED = 4
    ITR_COMPLETED = 5
    EXCEPTION_RAISED = -1

class Trainer(object):
    ''' Trainer\n
    Args:
        model (Module):
        optim (Optimizer):
        criterion (Function):
        scheduler ():
    '''
    def __init__(self, model, optim, criterion, scheduler=None):
        self.model = model
        self.optim = optim
        self.criterion = criterion
        self.scheduler = scheduler
        self._train_routine = None
        self._test_routine = None
        self.valid_events = []
        self._event_handlers = {}
        self._register_events(*Events)

    def __repr__(self):
        return '{}({}, {}, {}, {})'.format(self.__class__.__name__, self.model, self.optim, self.criterion, self.scheduler)

    def _register_events(self, *events):
        for e in events:
            self.valid_events.append(e)
    
    def add_event(self, event):
        self.valid_events.append(event)

    def add_handler(self, event, handler, *args, **kwargs):
        assert event in self.valid_events, logger.error('[*] invalid event was added.')
        if not event in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append((handler, args, kwargs))
    
    def event(self, event, *args, **kwargs):
        def decorator(f):
            self.add_handler(event, f, *args, **kwargs)
            return f
        return decorator

    def _on_event(self, event, *event_args, **event_kwargs):
        if event in self._event_handlers:
            for handler, args, kwargs in self._event_handlers[event]:
                handler(self, *args, *event_args, **kwargs, **event_kwargs)
    
    def set_train_routine(self, routine):
        self._train_routine = routine

    def set_test_routine(self, routine):
        self._test_routine = routine

    @property
    def train_routine(self):
        def decorator(f):
            self.set_train_routine(f)
            return f
        return decorator
    
    @property
    def test_routine(self):
        def decorator(f):
            self.set_test_routine(f)
            return f
        return decorator

    def train(self, dataloader, epochs=200):
        ''' helps the training process 
        Args: 
            dataloader (DataLoader): dataloader to use 
            epochs (int): number of epochs
        ''' 
        try:
            logger.info('[*] training started with max epochs of {}'.format(epochs))
            start = time.time()
            self._on_event(Events.STARTED)
            for epoch in range(epochs):
                self._on_event(Events.EPOCH_STARTED)
                train_loss = 0
                for i, (data, label) in enumerate(dataloader): 
                    self._on_event(Events.ITR_STARTED)
                    train_loss += self._train_routine(self, data, label)
                    self._on_event(Events.ITR_COMPLETED)
                    progressbar(i, len(dataloader), 'epoch: {}/{} train loss:{:.4f} '.format(epoch+1, epochs, train_loss/(i+1)), '(time: {})'.format(str(timedelta(seconds=time.time()-start))))
                logger.debug('epoch: {}/{} train loss:{:.4f} '.format(epoch+1, epochs, train_loss/len(dataloader)))
                self._on_event(Events.EPOCH_COMPLETED, epoch=epoch)
            self._on_event(Events.COMPLETED)
        except BaseException as e:
            logger.error('[*] training terminated due to exception: {}'.format(str(e)))
            self._on_event(Events.EXCEPTION_RAISED, e)        

    def test(self, dataloader, weights=None):
        ''' helps the testing process
        Args: 
            dataloader (DataLoader): dataloader to use 
            weights (str): path to pretrained weights
        ''' 
        self.model.eval()
        if weights is not None:
            if os.path.exists(filename):
                self.model.load(filename)
            else:
                logger.error('[*] file not found: {}'.format(weights))
                raise FileNotFoundError
        try:
            logger.info('[*] testing started.')
            start = time.time()
            self._on_event(Events.STARTED)
            for i, (data, label) in enumerate(dataloader): 
                self._on_event(Events.ITR_STARTED)
                self._test_routine(self, data, label)
                self._on_event(Events.ITR_COMPLETED)
                progressbar(i, len(dataloader), )
            self._on_event(Events.COMPLETED)
        except BaseException as e:
            logger.error('[*] training terminated due to exception: {}'.format(str(e)))
            self._on_event(Events.EXCEPTION_RAISED, e)    

    def get_accuracy(self, data, label):
        output = self.model(data) 
        out = np.argmax(output.data, axis=1) 
        ans = np.argmax(label.data, axis=1)
        return sum(out == ans)/label.shape[0]

    def plot(self, losses, filename=None):
        plt.plot([i for i in range(len(losses))], losses)
        plt.title('training losses of {} in {}'.format(self.model_name, self.data_name))
        plt.ylabel('training loss')
        plt.xlabel('epochs')
        if filename is not None:
            plt.savefig(filename)
        else: 
            plt.show()
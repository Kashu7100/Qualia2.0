# -*- coding: utf-8 -*- 
from ...core import *
import time
from datetime import timedelta
from enum import Enum
import matplotlib.pyplot as plt
from logging import getLogger
logger = getLogger('QualiaLogger').getChild('helper')

class Events(Enum):
    STARTED = "started"
    COMPLETED = "completed"
    EPOCH_STARTED = "epoch_started"
    EPOCH_COMPLETED = "epoch_completed"
    ITR_STARTED = "itr_started"
    ITR_COMPLETED = "itr_completed"
    EXCEPTION_RAISED = "exception_raised"

class Trainer(object):
    ''' Trainer\n
    '''
    def __init__(self, model, optim, criterion, scheduler=None):
        self.model = model
        self.optim = optim
        self.criterion = criterion
        self.scheduler = scheduler
        self._train_routine = None
        self.valid_events = []
        self._event_handlers = {}
        self._register_events(*Events)

    def __repr__(self):
        return '{}({}, {}, {}, {})'.format(self.__class__.__name__, self.model, self.optim, self.criterion, self.scheduler)

    def _register_events(self, *events):
        for e in events:
            self.valid_events.append(e)

    def add_event(self, event, handler, *args, **kwargs):
        assert event in self.valid_events, logger.error('[*] invalid event was added.')
        if not event in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append((handler, args, kwargs))
    
    def event(self, event_name, *args, **kwargs):
        def decorator(f):
            self.add_event(event_name, f, *args, **kwargs)
            return f
        return decorator

    def _on_event(self, event, *event_args, **event_kwargs):
        if event in self._event_handlers:
            for handler, args, kwargs in self._event_handlers[event_name]:
                handler(self, *args, *event_args, **kwargs, **event_kwargs)
    
    def set_train_routine(self, routine):
        self._train_routine = routine

    @property
    def train_routine(self):
        def decorator(f):
            self.set_train_routine(f)
            return f
        return decorator

    def train(self, train_loader, max_epochs=200):
        ''' trainer helps the training process of supervised learning
        Args: 
            train_loader (DataLoader): dataloader to use 
            max_epochs (int): number of epochs
        ''' 
        try:
            logger.info('[*] training started with max epochs of {}'.format(max_epochs))
            start = time.time()
            self._on_event(Events.STARTED)
            for epoch in range(epochs):
                self._on_event(Events.EPOCH_STARTED)
                for i, (data, label) in enumerate(dataloader): 
                    self._on_event(Events.ITR_STARTED)
                    loss = self._train_routine(self, data, label)
                    self._on_event(Events.ITR_COMPLETED)
                    progressbar(i, len(dataloader), 'epoch: {}/{} train loss:{:.4f} '.format(epoch+1, epochs, loss.asnumpy()), '(time: {})'.format(str(timedelta(seconds=time.time()-start))))
                logger.debug('epoch: {}/{} train loss:{:.4f} '.format(epoch+1, epochs, np.mean(tmp_loss)))
                self._on_event(Events.EPOCH_COMPLETED, epoch)
            self._on_event(Events.COMPLETED)
        except BaseException as e:
            logger.error('[*] training terminated due to exception: {}'.format(str(e)))
            self._on_event(Events.EXCEPTION_RAISED, e)        

    #TODO
    def test(self, test_loader, weights=None):
        self.model.eval()
        if weights is not None:
            if os.path.exists(filename):
                self.model.load(filename)
            else:
                logger.error('[*] file not found: {}'.format(weights))
                raise FileNotFoundError
        try:
            logger.info('[*] testing started.')
            acc = 0
            for i, (data, label) in enumerate(dataloader): 
                output = self.model(data) 
                out = np.argmax(output.data, axis=1) 
                ans = np.argmax(label.data, axis=1)
                acc += sum(out == ans)/label.shape[0]
                progressbar(i, len(dataloader))
            logger.info('\n[*] test acc: {:.2f}%'.format(float(acc/len(dataloader)*100)))
        except BaseException as e:
            logger.error('[*] training terminated due to exception: {}'.format(str(e)))
            self._on_event(Events.EXCEPTION_RAISED, e)      

    #TODO
    def plot(self, filename=None):
        plt.plot([i for i in range(len(self.losses))], self.losses)
        plt.title('training losses of {} in {}'.format(self.model_name, self.data_name))
        plt.ylabel('training loss')
        plt.xlabel('epochs')
        if filename is not None:
            plt.savefig(filename)
        else: 
            plt.show()
# -*- coding: utf-8 -*- 
import qualia2

import glob
import os
from logging import getLogger
logger = getLogger('QualiaLogger').getChild('checkpointer')

class Checkpointer(object):
    def __init__(self, model, optimizer=None, scheduler=None, save_period=50, save_dir=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_period = save_period
        self.save_dir = save_dir

    def __call__(self, trainer, *args, **kwargs):
        if kwargs['epoch'] % checkpoint.save_period == 0 and kwargs['epoch'] > 0:
            self.save('checkpoint_epoch{}'.format(kwargs['epoch']))

    def save(self, name, **kwargs):
        if not self.save_dir:
            return
        data = {}
        data['model'] = self.model.state_dict()
        if self.optimizer is not None:
            data['optimizer'] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data['scheduler'] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, name)
        self.logger.info('[*] saving checkpoint to {}'.format(save_file))
        qualia2.save(data, save_file)

    def load(self, file=None):
        if file is None:
            files = glob.glob(self.save_dir+'/*.qla')
            files.sort()
            if files:
                logger.info('[*] loading checkpoint from {}'.format(files[-1]))
                checkpoint = qualia2.load(files[-1])
            else:
                logger.error('[*] no checkpoint file was provided.')
                raise FileNotFoundError
        else:
            logger.info('[*] loading checkpoint from {}'.format(file))
            checkpoint = qualia2.load(file)
        self.model.load_state_dict(checkpoint.pop('model'))
        if 'optimizer' in checkpoint and self.optimizer:
            logger.info('[*] loading optimizer from {}'.format(file))
            self.optimizer.load_state_dict(checkpoint.pop('optimizer'))
        if 'scheduler' in checkpoint and self.scheduler:
            logger.info('[*] loading scheduler from {}'.format(file))
            self.scheduler.load_state_dict(checkpoint.pop('scheduler'))
        return checkpoint
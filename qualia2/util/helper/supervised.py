# -*- coding: utf-8 -*- 
from .helper import Events, Trainer

def supervised_trainer(model, optim, criterion, scheduler=None, checkpoint=None):
    trainer = Trainer(model, optim, criterion, scheduler)

    @trainer.train_routine
    def routine(trainer, data, label):
        trainer.model.train()
        trainer.optim.zero_grad()
        output = trainer.model(data) 
        loss = trainer.criterion(output, label)
        loss.backward()
        trainer.optim.step()
        return loss.asnumpy()
    
    if scheduler is not None:
        @trainer.event(Events.EPOCH_COMPLETED)
        def _scheduler(trainer, epoch):
            trainer.scheduler.step()

    if checkpoint is not None:
        trainer.add_handler(Events.EPOCH_COMPLETED, checkpoint)

    return trainer
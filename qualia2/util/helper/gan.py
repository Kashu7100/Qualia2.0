# -*- coding: utf-8 -*- 
from .helper import Events, Trainer

def gan_trainer(model, optim, criterion, scheduler=None):
    trainer = Trainer(model, optim, criterion, scheduler)

    @trainer.train_routine
    def routine(trainer, data, label):
        discriminator.train()
        generator.train()
        noise = Tensor(np.random.randn(self.batch, self.z_dim))
        fake_img = generator(noise)
        # update Discriminator
        # feed fake images
        output_fake = discriminator(fake_img.detach())
        loss_d_fake = criteria(output_fake, target_fake)
        # feed real images
        output_real = discriminator(data)
        loss_d_real = criteria(output_real, target_real*(1-smooth))
        loss_d = loss_d_fake + loss_d_real
        discriminator.zero_grad()
        loss_d.backward()
        optim_d.step()
        # update Generator
        discriminator.eval()
        output = discriminator(fake_img)
        loss_g = criteria(output, target_real)
        generator.zero_grad()
        loss_g.backward()
        optim_g.step()

    return trainer

def conditional_gan_trainer(model, optim, criterion, scheduler=None):
    pass

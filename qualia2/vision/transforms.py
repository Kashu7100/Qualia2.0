# -*- coding: utf-8 -*-
from ..autograd import Tensor
import PIL
import numpy 

class Compose(object):
    ''' Compose\n
    Composes several transforms together.
    Args:
        transforms (list): list of transforms
    '''
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

class Resize(object):
    ''' Resize\n
    Resize the input PIL Image to the given size.
    Args:
        size (int): Desired output size. (size * height / width, size)
        interpolation (int): Desired interpolation. Default is `PIL.Image.BILINEAR`
    '''
    def __init__(self, size, interpolation=PIL.Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image):
        w, h = image.size
        return image.resize((self.size, int(self.size*h/w)), self.interpolation)

class CenterCrop(object):
    ''' CenterCrop\n
    Crops the given PIL Image at the center.
    Args:
        size (int): Desired output size of the crop.
    '''
    def __init__(self, size):
        self.size = size
    
    def __call__(self, image):
        w, h = image.size
        left = (w-self.size)//2
        right = w-((w-self.size)//2+(w-self.size)%2)
        up = (h-self.size)//2
        bottom = h-((h-self.size)//2+(h-self.size)%2)
        return image.crop((left, up, right, bottom))

class ToTensor(object):
    '''
    Convert a PIL Image to Tensor.
    '''
    def __call__(self, image):
        image = numpy.asarray(image)
        image = image.transpose(2,0,1)
        image = image.reshape(1,*image.shape) / 255
        return Tensor(image)

class ToPIL(object):
    '''
    Convert Tensor to PIL Image.
    '''
    def __call__(self, tensor):
        data = tensor.asnumpy()
        data = data[0].transpose(1,2,0)
        return PIL.Image.fromarray(data)
    
class Normalize(object):
    '''
    Normalize a tensor image with mean and standard deviation.
    '''
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
    
    def __call__(self, image):
        if image.shape[1] == 3:
            image.data[:,0] = (image.data[:,0]-self.mean[0])/self.std[0]
            image.data[:,1] = (image.data[:,1]-self.mean[1])/self.std[1]
            image.data[:,2] = (image.data[:,2]-self.mean[2])/self.std[2]
        else:
            image.data[:,0] = (image.data[:,0]-self.mean[0])/self.std[0]
        return image

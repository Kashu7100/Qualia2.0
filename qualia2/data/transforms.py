# -*- coding: utf-8 -*-
from ..core import *
from ..autograd import Tensor
import numpy 
try:
    import Image
except ImportError:
    from PIL import Image

class Compose(object):
    ''' Compose\n
    Composes several transforms together.
    Args:
        transforms (list): list of transforms
    '''
    def __init__(self, transforms=[]):
        assert isinstance(transforms, list), '[*] transforms needs to be list of transforms.'
        self.transforms = transforms

    def __repr__(self):
        return '{}(transforms={})'.format(self.__class__.__name__, self.transforms)
        
    def __str__(self):
        return self.__class__.__name__

    def __call__(self, img):
        if not self.transforms:
            return img
        for t in self.transforms:
            img = t(img)
        return img

    def append(self, transform):
        self.transforms.append(transform)

class Resize(object):
    ''' Resize\n
    Resize the input PIL Image to the given size.
    Args:
        size (int): Desired output size. (size * height / width, size)
        interpolation (int): Desired interpolation. Default is `PIL.Image.BILINEAR`
    '''
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __repr__(self):
        return '{}(size={}, interpolation={})'.format(self.__class__.__name__, self.size, self.interpolation)
        
    def __str__(self):
        return self.__class__.__name__

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
    
    def __repr__(self):
        return '{}(size={})'.format(self.__class__.__name__, self.size)
        
    def __str__(self):
        return self.__class__.__name__
    
    def __call__(self, image):
        w, h = image.size
        left = (w-self.size)//2
        right = w-((w-self.size)//2+(w-self.size)%2)
        up = (h-self.size)//2
        bottom = h-((h-self.size)//2+(h-self.size)%2)
        return image.crop((left, up, right, bottom))

class ToTensor(object):
    '''
    Convert a PIL Image or numpy array to Tensor.
    '''
    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
        
    def __str__(self):
        return self.__class__.__name__

    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = numpy.asarray(image)
            image = image.transpose(2,0,1)
            image = image.reshape(1,*image.shape) / 255
        elif isinstance(image, np.ndarray):
            image = image / 255
        else:
            raise TypeError
        return Tensor(image)

class ToPIL(object):
    '''
    Convert Tensor to PIL Image.
    '''
    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
        
    def __str__(self):
        return self.__class__.__name__

    def __call__(self, tensor):
        data = tensor.asnumpy()
        data = data[0].transpose(1,2,0)
        return Image.fromarray(data)
    
class Normalize(object):
    '''
    Normalize a tensor image with mean and standard deviation.
    '''
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
    
    def __repr__(self):
        return '{}(mean={}, std={})'.format(self.__class__.__name__, self.mean, self.std)
        
    def __str__(self):
        return self.__class__.__name__
    
    def __call__(self, image):
        if image.shape[1] == 3:
            image.data[:,0] = (image.data[:,0]-self.mean[0])/self.std[0]
            image.data[:,1] = (image.data[:,1]-self.mean[1])/self.std[1]
            image.data[:,2] = (image.data[:,2]-self.mean[2])/self.std[2]
        else:
            image.data[:,0] = (image.data[:,0]-self.mean[0])/self.std[0]
        return image

class RandomHorizontalFlip(object):
    '''
    Horizontally flip the given Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    '''
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        '''
        Args:
            img (PIL Image): Image to be flipped.
        '''
        if random.random() < self.p:
            if isinstance(img, np.ndarray):
                return img[:,:,:,::-1]
            elif isinstance(img, Image.Image):
                return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomVerticalFlip(object):
    '''
    Vertically flip the given Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    '''
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        '''
        Args:
            img (PIL Image): Image to be flipped.
        '''
        if random.random() < self.p:
            if isinstance(img, np.ndarray):
                return img[:,:,::-1,:]
            elif isinstance(img, Image.Image):
                return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class Flatten(object):
    '''
    Flatten a image
    '''
    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
        
    def __str__(self):
        return self.__class__.__name__

    def __call__(self, image):
        return image.reshape(image.shape[0],-1)
# -*- coding: utf-8 -*- 
from .. import to_cpu
from ..core import *
from .dataset import *
from .transforms import Compose, ToTensor, Normalize
import matplotlib.pyplot as plt
import gzip

class EMNIST(Dataset):
    '''EMNIST Dataset\n     
    
    Args:
        train (bool): if True, load training dataset
        transforms (transforms): transforms to apply on the features
        target_transforms (transforms): transforms to apply on the labels

    Shape: 
        - data: [N, 1, 28, 28]
    '''
    def __init__(self, train=True, 
                transforms=Compose([ToTensor(), Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])]), 
                target_transforms=None):
        super().__init__(train, transforms, target_transforms)
    
    def __len__(self):
        if self.train:
            return 697932 
        else:
            return 116323

    def state_dict(self):
        return {
            'label_map': emnist_labels
        }

    def prepare(self):
        files = {
            'train_data.gz':'https://www.dropbox.com/s/rz2bpnt59k4zy26/emnist_train_data.gz?dl=1', 
            'train_labels.gz':'https://www.dropbox.com/s/tjnpaz89x1xjwk3/emnist_train_labels.gz?dl=1', 
            'test_data.gz':'https://www.dropbox.com/s/0ngevukoflx8wkr/emnist_test_data.gz?dl=1', 
            'test_labels.gz':'https://www.dropbox.com/s/4zgdxnlcfr3h54x/emnist_test_labels.gz?dl=1' 
        }
        for filename, value in files.items():
            self._download(value, filename)
        if self.train:
            self.data = self._load_data(self.root+'/train_data.gz')
            self.label = EMNIST.to_one_hot(self._load_label(self.root+'/train_labels.gz'), 61)
        else:
            self.data = self._load_data(self.root+'/test_data.gz')
            self.label = EMNIST.to_one_hot(self._load_label(self.root+'/test_labels.gz'), 61)
    
    def _load_data(self, filename):
        with gzip.open(filename, 'rb') as file: 
            if gpu:
                import numpy
                data = np.asarray(numpy.frombuffer(file.read(), np.uint8, offset=16))
            else:
                data = np.frombuffer(file.read(), np.uint8, offset=16) 
        return np.transpose(data.reshape(-1,1,28,28), (0,1,3,2))

    def _load_label(self, filename):
        with gzip.open(filename, 'rb') as file: 
            if gpu:
                import numpy
                labels = np.asarray(numpy.frombuffer(file.read(), np.uint8, offset=8))
            else:
                labels = np.frombuffer(file.read(), np.uint8, offset=8) 
        return labels

    def show(self, row=10, col=10):
        H, W = 28, 28
        img = np.zeros((H*row, W*col))
        for r in range(row):
            for c in range(col):
                img[r*H:(r+1)*H, c*W:(c+1)*W] = self.data[random.randint(0, len(self.data)-1)].reshape(H,W)
        plt.imshow(to_cpu(img) if gpu else img, cmap='gray', interpolation='nearest') 
        plt.axis('off')
        plt.show()  

emnist_labels = {
    0: chr(48),
    1: chr(49),
    2: chr(50),
    3: chr(51),
    4: chr(52),
    5: chr(53),
    6: chr(54),
    7: chr(55),
    8: chr(56),
    9: chr(57),
    10: chr(65),
    11: chr(66),
    12: chr(67),
    13: chr(68),
    14: chr(69),
    15: chr(70),
    16: chr(71),
    17: chr(72),
    18: chr(73),
    19: chr(74),
    20: chr(75),
    21: chr(76),
    22: chr(77),
    23: chr(78),
    24: chr(79),
    25: chr(80),
    26: chr(81),
    27: chr(82),
    28: chr(83),
    29: chr(84),
    30: chr(85),
    31: chr(86),
    32: chr(87),
    33: chr(88),
    34: chr(89),
    35: chr(90),
    36: chr(97),
    37: chr(98),
    38: chr(99),
    39: chr(100),
    40: chr(101),
    41: chr(102),
    42: chr(103),
    43: chr(104),
    44: chr(105),
    45: chr(106),
    46: chr(107),
    47: chr(108),
    48: chr(109),
    49: chr(110),
    50: chr(111),
    51: chr(112),
    52: chr(113),
    53: chr(114),
    54: chr(115),
    55: chr(116),
    56: chr(117),
    57: chr(118),
    58: chr(119),
    59: chr(120),
    60: chr(121),
    61: chr(122)
}
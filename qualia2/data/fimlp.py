# -*- coding: utf-8 -*- 
from .. import to_cpu
from ..core import *
from .dataset import *
from .transforms import Compose, ToTensor, Normalize
import matplotlib.pyplot as plt

class FIMLP(Dataset):
    '''FIMLP Dataset\n     
    
    Args:
        train (bool): if True, load training dataset
        transforms (transforms): transforms to apply on the features
        target_transforms (transforms): transforms to apply on the labels
        
    Shape: 
        - data: [N, 1, 96, 96] if flatten [N, 96*96]

    Label:
        Only x and y of the eyes center, nose tip and mouth center will be used as a label by default due to its small number of missing values.
        Landmarks                   Missing
        left_eye_center_x              10
        left_eye_center_y              10
        right_eye_center_x             13
        right_eye_center_y             13
        # left_eye_inner_corner_x      4778
        # left_eye_inner_corner_y      4778
        # left_eye_outer_corner_x      4782
        # left_eye_outer_corner_y      4782
        # right_eye_inner_corner_x     4781
        # right_eye_inner_corner_y     4781
        # right_eye_outer_corner_x     4781
        # right_eye_outer_corner_y     4781
        # left_eyebrow_inner_end_x     4779
        # left_eyebrow_inner_end_y     4779
        # left_eyebrow_outer_end_x     4824
        # left_eyebrow_outer_end_y     4824
        # right_eyebrow_inner_end_x    4779
        # right_eyebrow_inner_end_y    4779
        # right_eyebrow_outer_end_x    4813
        # right_eyebrow_outer_end_y    4813
        nose_tip_x                      0
        nose_tip_y                      0
        # mouth_left_corner_x          4780
        # mouth_left_corner_y          4780
        # mouth_right_corner_x         4779
        # mouth_right_corner_y         4779
        # mouth_center_top_lip_x       4774
        # mouth_center_top_lip_y       4774
        mouth_center_bottom_lip_x      33
        mouth_center_bottom_lip_y      33
    '''
    def __init__(self, train=True, 
                transforms=Compose([ToTensor(), Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])]), 
                target_transforms=None):
        self.divide = 6000      
        super().__init__(train, transforms, target_transforms)
        
    def prepare(self):
        self._download()
        if self.train:
            self.label, _ = self._load_label(self.root)
            self.data, _ = self._load_data(self.root)
        else:
            _, self.label = self._load_label(self.root)
            _, self.data = self._load_data(self.root)
            
            
    def _download(self): 
        if not os.path.exists(self.root+'/'):  
            os.makedirs(self.root+'/') 
        else:
            return
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files('drgilermo/face-images-with-marked-landmark-points', path=self.root, unzip=True)
    
    def _load_data(self, path):
        if gpu:
            import numpy
            data = np.asarray(numpy.moveaxis(numpy.load(path+'/face_images.npz', 'r')['face_images'], -1, 0))
        else:
            data = np.moveaxis(np.load(path+'/face_images.npz', 'r')['face_images'], -1, 0) 
        data = data[self.data_mask]
        
        return data[:self.divide].reshape(-1,1,96,96), data[self.divide:].reshape(-1,1,96,96)

    def _load_label(self, path):
        import numpy
        data = numpy.genfromtxt(path+'/facial_keypoints.csv', delimiter=',', filling_values=0)[1:]
        if gpu:
            data = np.asarray(data)
        
        # select x and y of the eyes center, nose tip and mouth center
        label = np.zeros((7049,8))
        label[:,0:4] = data[:,0:4]
        label[:,4:6] = data[:,20:22]
        label[:,6:8] = data[:,28:]
        self.data_mask = np.all(label>0, axis=1)
        label = label[self.data_mask]
        return label[:self.divide]/96.0, label[self.divide:]/96.0

    def show(self, row=5, col=5):
        H, W = 96, 96
        img = np.zeros((H*row, W*col))
        label_x = np.zeros((row, col, 4))
        label_y = np.zeros((row, col, 4))
        for r in range(row):
            for c in range(col):
                idx = random.randint(0, len(self.data)-1)
                img[r*H:(r+1)*H, c*W:(c+1)*W] = self.data[idx].reshape(H,W)/255
                label_x[r:(r+1), c:(c+1)] = self.label[idx,0::2]*W + c*W
                label_y[r:(r+1), c:(c+1)] = self.label[idx,1::2]*H + r*H
        plt.imshow(to_cpu(img), cmap='gray', interpolation='nearest') 
        plt.scatter(to_cpu(label_x.reshape(-1)),to_cpu(label_y.reshape(-1)),marker='o',c='r',s=10)
        plt.axis('off')
        plt.show()
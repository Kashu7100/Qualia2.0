from .. import to_cpu
from ..core import *
from ..autograd import Tensor
from .dataloader import ImageLoader
import matplotlib.pyplot as plt
import os

class FIMLP(ImageLoader):
    '''FIMLP Dataset\n     
    Args:
        normalize (bool): If true, the intensity value of a specific pixel in a specific image will be rescaled from [0, 255] to [0, 1]. Default: True 
        flatten (bool): If true, data will have a shape of [N, 96*96]. Default: False 
    Shape: 
        - data: [N, 1, 96, 96] if flatten [N, 96*96]

    Label:
        Only x and y of the eyes center, nose tip and mouth center will be used as a label by default due to its small number of missing values.
        Landmarks                   Missing
        left_eye_center_x              10
        left_eye_center_y              10
        right_eye_center_x             13
        right_eye_center_y             13
        left_eye_inner_corner_x      4778
        left_eye_inner_corner_y      4778
        left_eye_outer_corner_x      4782
        left_eye_outer_corner_y      4782
        right_eye_inner_corner_x     4781
        right_eye_inner_corner_y     4781
        right_eye_outer_corner_x     4781
        right_eye_outer_corner_y     4781
        left_eyebrow_inner_end_x     4779
        left_eyebrow_inner_end_y     4779
        left_eyebrow_outer_end_x     4824
        left_eyebrow_outer_end_y     4824
        right_eyebrow_inner_end_x    4779
        right_eyebrow_inner_end_y    4779
        right_eyebrow_outer_end_x    4813
        right_eyebrow_outer_end_y    4813
        nose_tip_x                      0
        nose_tip_y                      0
        mouth_left_corner_x          4780
        mouth_left_corner_y          4780
        mouth_right_corner_x         4779
        mouth_right_corner_y         4779
        mouth_center_top_lip_x       4774
        mouth_center_top_lip_y       4774
        mouth_center_bottom_lip_x      33
        mouth_center_bottom_lip_y      33
    '''
    def __init__(self, normalize=True, flatten=False):
        super().__init__() 
        path = os.path.dirname(os.path.abspath(__file__)) 
        self.divide = 6000

        print('[*] preparing data...')
        if not os.path.exists(path + '/download/fimlp/'): 
            print('    this might take few minutes.') 
            self.download(path+'/download/')
        self.train_label, self.test_label = self._load_label(path+ '/download/fimlp/')
        self.train_data, self.test_data = self._load_data(path+ '/download/fimlp/')
        print('[*] done.')

        if normalize: 
            self.train_data = np.divide(self.train_data, 255.0)
            self.test_data = np.divide(self.test_data, 255.0)
        if flatten:
            self.train_data = self.train_data.reshape(-1, 96*96) 
            self.test_data = self.test_data.reshape(-1, 96*96) 

    def download(self, path): 
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files('drgilermo/face-images-with-marked-landmark-points', path=path+'fimlp', unzip=True)
    
    def _load_data(self, path):
        if gpu:
            import numpy
            data = np.asarray(numpy.moveaxis(numpy.load(path+'face_images.npz', 'r')['face_images'], -1, 0))
        else:
            data = np.moveaxis(np.load(path+'face_images.npz', 'r')['face_images'], -1, 0) 
        data = data[self.data_mask]
        
        return data[:self.divide].reshape(-1,1,96,96), data[self.divide:].reshape(-1,1,96,96)

    def _load_label(self, path):
        import numpy
        data = numpy.genfromtxt(path+'facial_keypoints.csv', delimiter=',', filling_values=0)[1:]
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

    def show(self):
        import random
        for i in range(16):
            plt.subplot(4,4,i+1)
            plt.xticks([]) 
            plt.yticks([]) 
            plt.grid(False)
            idx = random.choices(range(self.divide))
            img = self.train_data[idx].reshape(96,96)
            plt.imshow(to_cpu(img) if gpu else img, cmap='gray', interpolation='nearest') 
            if gpu:
                plt.scatter(to_cpu(self.train_label[idx,0::2]*96),to_cpu(self.train_label[idx,1::2]*96),marker='o',c='r',s=10)
            else:
                plt.scatter(self.train_label[idx,0::2]*96,self.train_label[idx,1::2]*96,marker='o',c='r',s=10)
        plt.show() 

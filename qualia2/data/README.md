## Datasets
### MNIST
The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image. 
<p align="center">
  <img src="/assets/mnist_data.png">
</p>

### FashionMNIST
Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. 
<p align="center">
  <img src="/assets/fashion_mnist_data.png">
</p>

### KuzushiMNIST
Kuzushiji-MNIST contains 70,000 28x28 grayscale images spanning 10 classes (one from each column of hiragana), and is perfectly balanced like the original MNIST dataset (6k/1k train/test for each class).

<p align="center">
  <img src="/assets/kuzushi_mnist_data.png">
</p>

### Kuzushi49
Kuzushiji-49 contains 270,912 images spanning 49 classes, and is an extension of the Kuzushiji-MNIST dataset.

<p align="center">
  <img src="/assets/kuzushi49_data.png">
</p>

### CIFAR10
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 

<p align="center">
  <img src="/assets/cifar10_data.png">
</p>

### CIFAR100
The CIFAR-100 dataset consists of 60000 32x32 colour images in 100 classes containing 600 images each. There are 500 training images and 100 testing images per class. The 100 classes in the CIFAR-100 are grouped into 20 superclasses. 

<p align="center">
  <img src="/assets/cifar100_data.png">
</p>

### STL-10
[[site]](https://cs.stanford.edu/~acoates/stl10/)
The STL-10 dataset is an image recognition dataset for developing unsupervised feature learning, deep learning, self-taught learning algorithms. It contains 500 training images (10 pre-defined folds) and 800 test images per class.

The following standardized testing protocol for reporting results is recommended:

* Perform unsupervised training on the unlabeled.
* Perform supervised training on the labeled data using 10 (pre-defined) folds of 100 examples from the training data. The indices of the examples to be used for each fold are provided.
* Report average accuracy on the full test set.

<p align="center">
  <img src="/assets/stl10_data.png">
</p>

### FIMLP
FIMLP (Face Images with Marked Landmark Points) is a Kaggle's Facial Keypoint Detection dataset that contains 7049 96x96 facial images and up to 15 keypoints marked on them. 

<p align="center">
  <img src="/assets/fimlp_data.png">
</p>

### ChestXRay
The Chest X-Ray Images dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal). 

<p align="center">
  <img src="/assets/chestxray_data.png">
</p>

### LFWcrop
LFWcrop is a cropped version of the Labeled Faces in the Wild (LFW) dataset. LFWcrop was created due to concern about the misuse of the original LFW dataset, where face matching accuracy can be unrealistically boosted through the use of background parts of images (i.e. exploitation of possible correlations between faces and backgrounds).

### EMNIST
The EMNIST dataset is a set of handwritten character digits derived from the NIST Special Database 19  and converted to a 28x28 pixel image format and dataset.

### [Speech Commands Dataset](https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz)
Speech Commands Dataset consists of over 105,000 WAVE audio files of people saying thirty different words. This data was collected by Google and released under a CC BY license.

### [DeepLesion](https://nihcc.app.box.com/v/DeepLesion/folder/50715173939)
NIH Clinical Center releases dataset of 32,000 CT images

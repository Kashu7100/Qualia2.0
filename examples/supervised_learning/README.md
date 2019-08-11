# Image classification

## Alexnet
[[paper]](https://arxiv.org/pdf/1404.5997.pdf)


<p align="center">
  <img src="/assets/alexnet_model.png"/>
  <br>
  <b> Fig.1: </b> The configurations of alexnet 
</p>

## VGG
[[paper]](https://arxiv.org/pdf/1409.1556.pdf)

<p align="center">
  <img src="/assets/vgg_config.png"/>
  <br>
  <b> Fig.2: </b> The configurations of vgg models 
</p>


## ResNet
[[paper]](https://arxiv.org/pdf/1512.03385.pdf)

A training of deeper neural networks are difficult because of the vanishing gradient problem in gradient-based learning methods and backpropagation. A residual learning framework ease the training of networks that are substantially deep. 

<p align="center">
  <img src="/assets/resnet_block.png"/>
  <br>
  <b> Fig.3: </b> Residual learning: a building block. 
</p>

# Pose estimation
Pose estimation refers to computer vision techniques that detect human figures in images and videos, so that one could determine, for example, where someone’s elbow shows up in an image.

<p align="center">
  <img src="/assets/openpose_skelton.png"/>
  <br>
  <b> Fig.4: </b> Parts and Pairs indexes for COCO dataset.
</p>

## OpenPose
[[paper]](https://arxiv.org/pdf/1812.08008.pdf) [[example]](/examples/supervised_learning/openpose)

OpenPose provides a real-time method for Multi-Person 2D Pose Estimation based on its bottom-up approach instead of detection-based approach.

<p align="center">
  <img src="/assets/openpose_structure.png"/>
  <br>
  <b> Fig.5: </b> Architecture  of  the  multi-stage  CNN.
</p>


<p align="center">
  <img src="/assets/heatmap_paf.png"/>
  <br>
  <b> Fig.6: </b> Body part detection and part association.
</p>

The feature maps obtained by the first 10 layers of VGG-19 model are processed with multiple stages CNN to generate a set of Part Confidence Maps and a set of Part Affinity Fields (PAFs). They are then used in a greedy algorithm to obtain the poses for each person in the image.

# Object Detection
## SSD

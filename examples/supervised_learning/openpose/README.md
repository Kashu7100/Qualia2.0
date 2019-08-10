# OpenPose  
## heatmaps and PAFs
Heatmaps and Part Affinity Fields (PAFs) are the output form the OpenPose. PAFs are used to connect the detected body parts for several people in a single image at the same time.

<p align="center">
  <img src="/assets/women.png"/ width=420>
  <img src="/assets/women_heatmap2.png"/ width=420>
  <img src="/assets/women_heatmap1.png"/ width=420>
  <img src="/assets/women_paf.png"/ width=420>
  <br>
  <b> Fig.1: </b> Input image (upper left). Heatmap of the left shoulder (upper right). 
  <br>Heatmap of the left elbow (bottom left). PAF of the upper arm. (bottom right)
</p>

## pose estimation

To run a pose estimation with the pretrained model: 

```bash
$ python demo.py
```

<p align="center">
  <img src="/assets/women_pose.png"/>
  <br>
  <b> Fig.2: </b> Constructed pose estimation on picture.
</p>


<p align="center">
  <img src="/assets/baseball.gif"/>
  <br>
  <b> Fig.3: </b> Constructed pose estimation on video.
</p>

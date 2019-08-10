from qualia2.vision import OpenPoseBody
import qualia2.vision.transforms as transforms
from util import decode_pose
import PIL
import numpy
import matplotlib.pyplot as plt
from matplotlib.cm import jet
import cv2

img = PIL.Image.open('./women.jpg')

preprocess = transforms.Compose([
   transforms.Resize((368,368)),
   transforms.ToTensor(),
   transforms.Normalize([0.5,0.5,0.5],[1,1,1])
])

input = preprocess(img)

model = OpenPoseBody(pretrained=True)
model.eval()

pafs, heatmaps = model(input)

pafs = pafs.asnumpy()[0].transpose(1,2,0)
heatmaps = heatmaps.asnumpy()[0].transpose(1,2,0)

pafs = cv2.resize(pafs, (368,368), interpolation=cv2.INTER_CUBIC)
heatmaps = cv2.resize(heatmaps, (368,368), interpolation=cv2.INTER_CUBIC)

w, h = img.size

pafs = cv2.resize(pafs, (w, h), interpolation=cv2.INTER_CUBIC)
heatmaps = cv2.resize(heatmaps, (w, h), interpolation=cv2.INTER_CUBIC)

_, result, _, _ = decode_pose(numpy.asarray(img), heatmaps, pafs)

plt.imshow(result)
plt.axis('off')
plt.show()

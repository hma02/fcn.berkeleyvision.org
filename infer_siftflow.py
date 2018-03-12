import numpy as np
from PIL import Image

import caffe
import vis

# the demo image is "2007_000129" from PASCAL VOC

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
im = Image.open('demo/image.jpg')
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ -= np.array((114.578, 115.294, 108.353))
in_ = in_.transpose((2,0,1))

import scipy.io
label = scipy.io.loadmat('demo/label.mat')['S']
label = label.astype(np.uint8)
# label -= 1  # rotate labels so classes start at 0, void is 255
# label = label[np.newaxis, ...]

from scipy.misc import imsave
imsave('demo/label.png',label)

# load net
net = caffe.Net('siftflow-fcn8s/deploy.prototxt', 'siftflow-fcn8s/siftflow-fcn8s-heavy.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_
# run net and take argmax for prediction
net.forward()
out = net.blobs['score'].data[0].argmax(axis=0)

# visualize segmentation in PASCAL VOC colors
voc_palette = vis.make_palette(33)
out_im = Image.fromarray(vis.color_seg(out, voc_palette))
out_im.save('demo/output.png')
masked_im = Image.fromarray(vis.vis_seg(im, out, voc_palette))
masked_im.save('demo/visualization.jpg')

import caffe

import numpy as np
from PIL import Image
import scipy.io

import random

class SIFTFlowSegDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from SIFT Flow
    one-at-a-time while reshaping the net to preserve dimensions.

    This data layer has three tops:

    1. the data, pre-processed
    2. the semantic labels 0-32 and void 255
    # 3. the geometric labels 0-2 and void 255

    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - siftflow_dir: path to SIFT Flow dir
        - split: train / val / test
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for semantic segmentation of object and geometric classes.

        example: params = dict(siftflow_dir="/path/to/siftflow", split="val")
        """
        # config
        params = eval(self.param_str)
        self.siftflow_dir = params['siftflow_dir']
        self.batch_size = params['batch_size']
        self.height, self.width = 128,128
        # self.split = params['split']
        # self.mean = np.array((114.578, 115.294, 108.353), dtype=np.float32)
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)

        # two tops: data, semantic
        if len(top) != 2:
            raise Exception("Need to define three tops: data, semantic label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        # split_f  = '{}/{}.txt'.format(self.siftflow_dir, self.split)
        
        # self.indices = open(split_f, 'r').read().splitlines()
        import glob
        self.indices = sorted([s.split('/')[-1].split('.jpg')[0] for s in glob.glob(self.siftflow_dir+'/*.jpg')])
        self.idx = 0

        # make eval deterministic
        # if 'train' not in self.split:
        self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)

    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = np.zeros((self.batch_size, 3, self.height, self.width), np.float32)
        self.label_semantic = np.zeros((self.batch_size, 1, self.height, self.width), np.float32)
        for i in range(self.batch_size):
            # self.data = self.load_image(self.indices[self.idx+i])
            self.data[i, :, :, :] = self.load_image(self.indices[self.idx+i])
            self.label_semantic[i, :, :, :] = self.load_label(self.indices[self.idx+i], label_type='semantic')
        
        # print self.batch_size
        # self.label_geometric = self.load_label(self.indices[self.idx], label_type='geometric')
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(*self.data.shape)
        top[1].reshape(*self.label_semantic.shape)
        # top[2].reshape(1, *self.label_geometric.shape)

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label_semantic
        # top[2].data[...] = self.label_geometric

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += self.batch_size
            if self.idx + self.batch_size >= len(self.indices):
                self.idx = 0

    def backward(self, top, propagate_down, bottom):
        pass

    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open('{}/{}.jpg'.format(self.siftflow_dir, idx))
        
        im = im.resize([self.height, self.width], Image.ANTIALIAS)
        
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        # in_ -= self.mean
        in_ = in_.transpose((2,0,1))
        
        in_ = in_ /255.
        
        return in_

    def load_label(self, idx, label_type=None):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        if label_type == 'semantic':
            label = Image.open('{}/{}.jpg.png'.format(self.siftflow_dir, idx))
        else:
            raise Exception("Unknown label type")
        # label = label.astype(np.uint8)
        # label -= 1  # rotate labels so classes start at 0, void is 255
        # label = label[np.newaxis, ...]
        
        Dtype = np.uint8
        
        
        # L = np.array(label)
        # Limg = Image.fromarray(L)
        label = label.resize([self.height, self.width],Image.NEAREST) # To resize the Label file to the required size 
        L = np.array(label,Dtype)
        
        # print L.shape
        
        L = (L > 200) * 1
        
        if len(L.shape)==3:
            # print 3, L.mean()
            L = L[:,:,0][:,:,np.newaxis] # when it is close to black or close to white, the RGB pixels are all very large or all very small, so taking one of them is close to taking the mean of them
        elif len(L.shape)==2:
            # print 2, L.mean()
            L= L[:,:,np.newaxis]
        else:
            raise ValueError('unexpected dimension')
            
        # L = L.reshape(L.shape[0],L.shape[1],1)
        
        # print L.shape

        label = L.transpose((2,0,1)).astype(Dtype)
        
        return label.copy()

import numpy as np
from PIL import Image

import caffe
import vis

# load net
net = caffe.Net('hair-mn/deploy.prototxt', 'hair-mn/mobilenet_iter_110000.caffemodel', caffe.TEST)
    
def infer(input_image,input_mask,input_mask_thr,output_mask,output_overlay):
    # the demo image is "2007_000129" from PASCAL VOC

    # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    image = Image.open(input_image)
    image = image.resize([128, 128], Image.ANTIALIAS)
    in_ = np.array(image, dtype=np.float32)
    in_ = in_[:,:,::-1]
    # in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_.transpose((2,0,1))


    label = Image.open(input_mask)
    label = label.resize([128, 128],Image.NEAREST)
    label = np.array(label,np.uint8)
    # label -= 1  # rotate labels so classes start at 0, void is 255
    # label = label[np.newaxis, ...]
    label = (label > 200) * 1
    from scipy.misc import imsave
    if input_mask_thr: imsave(input_mask_thr,label)

    
    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    net.forward()
    out = net.blobs['output_sep'].data[0]

    Dtype = out.dtype
    im = Image.fromarray(out[0,:,:])
    im = im.resize([6, 6], Image.NEAREST)
    im = np.array(im,Dtype)
    print im

    out = out.argmax(axis=0)

    im = Image.fromarray(out.astype(Dtype))
    im = im.resize([6, 6], Image.NEAREST)
    im = np.array(im,Dtype)
    print im

    # visualize segmentation in PASCAL VOC colors
    voc_palette = vis.make_palette(2)
    out_im = Image.fromarray(vis.color_seg(out, voc_palette))
    out_im.save(output_mask)
    masked_im = Image.fromarray(vis.vis_seg(image, out, voc_palette))
    masked_im.save(output_overlay)
    
import glob

image_files = sorted(glob.glob('demo/*.jpg'))
mask_files = sorted(glob.glob('demo/*.png'))

for input_image, input_mask in zip(image_files,mask_files):
    
    # input_mask_thr = input_mask.split('.jpg')[0]+'_thr.png'
    
    output_mask = input_mask.split('.jpg')[0]+'_output.png'
    
    output_overlay = input_image.split('.jpg')[0]+'_overlay.jpg'
    
    infer(input_image,input_mask,None,output_mask,output_overlay)
    

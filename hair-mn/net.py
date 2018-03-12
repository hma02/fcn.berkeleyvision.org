import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop

def conv_relu(bottom, nout, ks=3, stride=2, pad=1):
    conv = L.Convolution(bottom, 
        param=[dict(lr_mult=1, decay_mult=1)],
        convolution_param=dict(
                                kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad,
                                bias_term=False,
                                weight_filler=dict(type="msra"),
                                engine= 1 # DEFAULT = 0; CAFFE = 1; CUDNN = 2;
                                ) 
        )
    return conv, L.ReLU(conv, in_place=True)

# def max_pool(bottom, ks=2, stride=2):
#     return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


def upsample(bottom, nout):
    
    return L.Deconvolution(bottom, 
        param=[dict(lr_mult=0, decay_mult=0)],
        convolution_param=dict(
                                kernel_size=2, 
                                stride=2,
                                num_output=nout,
                                bias_term=False,
                                weight_filler=dict(type="bilinear"),
                                engine= 1 # DEFAULT = 0; CAFFE = 1; CUDNN = 2;
                                )
        )
            
        
def skip(bottom, nin, nout, ks):
    
    conv = L.Convolution(bottom, 
        param=[dict(lr_mult=1, decay_mult=1)],
        convolution_param=dict(
                            kernel_size=ks, stride=1,
                            num_output=nout, pad=0,
                            bias_term=False,
                            weight_filler=dict(type="msra"),
                            engine= 1 # DEFAULT = 0; CAFFE = 1; CUDNN = 2;
                            )
        )
        
    return conv

def eltsum(bottom1, bottom2):
    
    return L.Eltwise(bottom1, bottom2, operation=P.Eltwise.SUM)

def conv_dw(bottom, nin, nout, stride):
    
    conv_dw = L.Convolution(bottom, 
        param=[dict(lr_mult=1, decay_mult=1)],
        convolution_param=dict(
                                num_output=nin, kernel_size=3, stride=stride, pad=1, group=nin,
                                bias_term=False,
                                weight_filler=dict(type="msra"),
                                engine= 1 # DEFAULT = 0; CAFFE = 1; CUDNN = 2;
                                )
        )
        
    conv_sep = L.Convolution(conv_dw, 
        param=[dict(lr_mult=1, decay_mult=1)],
        convolution_param=dict(
                            num_output=nout, kernel_size=1, stride=1, pad=0,
                            bias_term=False,
                            weight_filler=dict(type="msra"),
                            engine= 1 # DEFAULT = 0; CAFFE = 1; CUDNN = 2;
                            )
        )
    
    return conv_dw, conv_sep, L.ReLU(conv_sep, in_place=True)

class MobileNetHair():
    def __init__(self, split):
        
        if split!='test':
            self.split='train'
        else:
            self.split='test'
        
    def forward(self):
        
        n = caffe.NetSpec()
        
        n.data, n.sem = L.Python(module='hair_layers',
                layer='SIFTFlowSegDataLayer', ntop=2,
                param_str=str(dict(siftflow_dir='../data/hair/realdata/'+self.split,
                    split=self.split, seed=1337, batch_size=4)))
        
        n.initial_conv, n.initial_relu = conv_relu(n.data, 16, 3, 2, 1)
        
        n.conv_dw_1 , n.conv_sep_1 , n.relu_1  = conv_dw(n.initial_relu, 16, 32, 1)
        n.conv_dw_2 , n.conv_sep_2 , n.relu_2  = conv_dw(n.relu_1      , 32, 64, 2)
        n.conv_dw_3 , n.conv_sep_3 , n.relu_3  = conv_dw(n.relu_2      , 64, 64, 1)
        n.conv_dw_4 , n.conv_sep_4 , n.relu_4  = conv_dw(n.relu_3      , 64, 128, 2)
        n.conv_dw_5 , n.conv_sep_5 , n.relu_5  = conv_dw(n.relu_4      , 128, 128, 1)
        n.conv_dw_6 , n.conv_sep_6 , n.relu_6  = conv_dw(n.relu_5      , 128, 256, 2)
        n.conv_dw_7 , n.conv_sep_7 , n.relu_7  = conv_dw(n.relu_6      , 256, 256, 1)
        n.conv_dw_8 , n.conv_sep_8 , n.relu_8  = conv_dw(n.relu_7      , 256, 256, 1)
        n.conv_dw_9 , n.conv_sep_9 , n.relu_9  = conv_dw(n.relu_8      , 256, 256, 1)
        n.conv_dw_10, n.conv_sep_10, n.relu_10 = conv_dw(n.relu_9      , 256, 256, 1)
        n.conv_dw_11, n.conv_sep_11, n.relu_11 = conv_dw(n.relu_10     , 256, 256, 1)
        n.conv_dw_12, n.conv_sep_12, n.relu_12 = conv_dw(n.relu_11     , 256, 512, 2)
        n.conv_dw_13, n.conv_sep_13, n.relu_13 = conv_dw(n.relu_12     , 512, 512, 1)
        
        
        n.up1 = upsample(n.relu_13, 512)
        n.skip1 = skip(n.relu_11, 256, 512, 1)
        n.up1 = eltsum(n.up1,n.skip1)
        n.filt1_dw, n.filt1_sep, n.filt1 = conv_dw(n.up1, 512, 16, 1)
        
        n.up2 = upsample(n.filt1, 16)
        n.skip2 = skip(n.relu_5, 128, 16, 1)
        n.up2 = eltsum(n.up2,n.skip2)
        n.filt2_dw, n.filt2_sep, n.filt2 = conv_dw(n.up2, 16, 16, 1)
        
        n.up3 = upsample(n.filt2, 16)
        n.skip3 = skip(n.relu_3, 64, 16, 1)
        n.up3 = eltsum(n.up3,n.skip3)
        n.filt3_dw, n.filt3_sep, n.filt3 = conv_dw(n.up3, 16, 16, 1)
        
        n.up4 = upsample(n.filt3, 16)
        n.skip4 = skip(n.relu_1, 32, 16, 1)
        n.up4 = eltsum(n.up4,n.skip4)
        n.filt4_dw, n.filt4_sep, n.filt4 = conv_dw(n.up4, 16, 16, 1)
        
        n.up5 = upsample(n.filt4, 16)
        n.filt5_dw, n.filt5_sep, n.filt5 = conv_dw(n.up5, 16, 16, 1)
        
        n.output_dw, n.output_sep, n.output = conv_dw(n.filt5, 16, 2, 1)
        
        n.loss = L.SoftmaxWithLoss(n.output, n.sem,
                loss_param=dict(normalize=False))
        
        return n.to_proto()

def make_net():
    with open('trainval.prototxt', 'w') as f:
        f.write(str(MobileNetHair('trainval').forward()))

    with open('test.prototxt', 'w') as f:
        f.write(str(MobileNetHair('test').forward()))

if __name__ == '__main__':
    make_net()

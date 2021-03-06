import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop

def conv_relu(bottom, nout, ks=3, stride=2, pad=1, bn=True, test=False):
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
    if bn:
        conv = L.BatchNorm(conv, 
                             # param=[dict(lr_mult=0, decay_mult=0),
                             #        dict(lr_mult=0, decay_mult=0),
                             #        dict(lr_mult=0, decay_mult=0)],
                             batch_norm_param=dict(
                                use_global_stats=test,
                                eps=0.00001
                             ),
                             # inplace=True
                            )
        conv = L.Scale(conv,
                             # param=[dict(lr_mult=0, decay_mult=0),
                             #        dict(lr_mult=0, decay_mult=0)
                             #        ],
                             scale_param = dict(
                                filler=dict(value=1),
                                bias_term=True,
                                bias_filler=dict(value=0)
                                            ),
                              # inplace=True
                             )
    relu_conv = L.ReLU(conv, in_place=True)
    return conv, relu_conv

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

def conv_dw(bottom, nin, nout, stride, bn=True, test=False):
    
    conv_dw = L.Convolution(bottom, 
        param=[dict(lr_mult=1, decay_mult=1)],
        convolution_param=dict(
                                num_output=nin, kernel_size=3, stride=stride, pad=1, group=nin,
                                bias_term=False,
                                weight_filler=dict(type="msra"),
                                engine= 1 # DEFAULT = 0; CAFFE = 1; CUDNN = 2;
                                )
        )
        
    if bn:
        conv_dw = L.BatchNorm(conv_dw, 
                             # param=[dict(lr_mult=0, decay_mult=0),
                             #        dict(lr_mult=0, decay_mult=0),
                             #        dict(lr_mult=0, decay_mult=0)],
                             batch_norm_param=dict(
                                use_global_stats=test,
                                eps=0.00001
                             ),
                             # inplace=True
                            )
        conv_dw = L.Scale(conv_dw,
                             # param=[dict(lr_mult=0, decay_mult=0),
                             #        dict(lr_mult=0, decay_mult=0)
                             #        ],
                             scale_param = dict(
                                filler=dict(value=1),
                                bias_term=True,
                                bias_filler=dict(value=0)
                                            ),
                             # inplace=True
                             )
        conv_dw = L.ReLU(conv_dw, in_place=True)
    
        
    conv_sep = L.Convolution(conv_dw, 
        param=[dict(lr_mult=1, decay_mult=1)],
        convolution_param=dict(
                            num_output=nout, kernel_size=1, stride=1, pad=0,
                            bias_term=False,
                            weight_filler=dict(type="msra"),
                            engine= 1 # DEFAULT = 0; CAFFE = 1; CUDNN = 2;
                            )
        )
    
    if bn:
        conv_sep = L.BatchNorm(conv_sep, 
                             # param=[dict(lr_mult=0, decay_mult=0),
                             #        dict(lr_mult=0, decay_mult=0),
                             #        dict(lr_mult=0, decay_mult=0)],
                             batch_norm_param=dict(
                                use_global_stats=test,
                                eps=0.00001
                             ),
                             # inplace=True
                            )
        conv_sep = L.Scale(conv_sep,
                             # param=[dict(lr_mult=0, decay_mult=0),
                             #        dict(lr_mult=0, decay_mult=0)
                             #        ],
                             scale_param = dict(
                                filler=dict(value=1),
                                bias_term=True,
                                bias_filler=dict(value=0)
                                            ),
                              # inplace=True
                             )
                         
    relu_conv = L.ReLU(conv_sep, in_place=True)
    
    return conv_dw, conv_sep, relu_conv

class MobileNetHair():
    def __init__(self, split):
        
        if split!='test':
            self.split='train'
            self.batch_size=4
            self.lmdb_data_path = '/home/allanma/data/hairlmdb/realdata/train_data'
            self.lmdb_label_path = '/home/allanma/data/hairlmdb/realdata/train_label'
            self.test=False
        else:
            self.split='test'
            self.batch_size=1
            self.lmdb_data_path = '/home/allanma/data/hairlmdb/realdata/test_data'
            self.lmdb_label_path = '/home/allanma/data/hairlmdb/realdata/test_label'
            self.test=True
        
        
    def forward(self):
        
        n = caffe.NetSpec()
        
        # n.data, n.label = L.Python(module='hair_layers',
        #         layer='SIFTFlowSegDataLayer', ntop=2,
        #         param_str=str(dict(siftflow_dir='../data/hair/realdata/'+self.split,
        #             split=self.split, seed=1337, batch_size=self.batch_size)))
          
        n.data = L.Data( 
                                transform_param=dict(
                                                      scale=0.00390625,
                                                      # mirror=true,
                                                      # crop_size=100,
                                                    #   mean_value: [103.94,116.78,123.68]
                                                        ),
                                data_param=dict(
                                                batch_size = self.batch_size,
                                                backend = caffe.params.Data.LMDB, 
                                                source = self.lmdb_data_path, 
                                                ),
                                                ntop = 1
                                                
                                )
        n.label = L.Data( 
                                transform_param=dict(
                                                      # scale=0.00390625,
                                                      # mirror=true,
                                                      # crop_size=100,
                                                    #   mean_value: [103.94,116.78,123.68]
                                                        ),
                                data_param=dict(
                                                batch_size = self.batch_size,
                                                backend = caffe.params.Data.LMDB, 
                                                source = self.lmdb_label_path, 
                                                ),
                                                ntop = 1
                                )
        # net.data_aug = caffe.layers.Python(net.data,
        #                                        python_param = dict(module = 'tools.layers', layer = 'DataAugmentationRandomMultiplicativeNoiseLayer'))
        # net.labels_aug = caffe.layers.Python(net.labels,
        #                                          python_param = dict(module = 'tools.layers', layer = 'DataAugmentationDuplicateLabelsLayer'))
        
        n.initial_conv, n.initial_relu = conv_relu(n.data, 16, 3, 2, 1, test=self.test)
        
        n.conv_dw_1 , n.conv_sep_1 , n.relu_1  = conv_dw(n.initial_relu, 16, 32, 1  , test=self.test)
        n.conv_dw_2 , n.conv_sep_2 , n.relu_2  = conv_dw(n.relu_1      , 32, 64, 2  , test=self.test)
        n.conv_dw_3 , n.conv_sep_3 , n.relu_3  = conv_dw(n.relu_2      , 64, 64, 1  , test=self.test)
        n.conv_dw_4 , n.conv_sep_4 , n.relu_4  = conv_dw(n.relu_3      , 64, 128, 2 , test=self.test)
        n.conv_dw_5 , n.conv_sep_5 , n.relu_5  = conv_dw(n.relu_4      , 128, 128, 1, test=self.test)
        n.conv_dw_6 , n.conv_sep_6 , n.relu_6  = conv_dw(n.relu_5      , 128, 256, 2, test=self.test)
        n.conv_dw_7 , n.conv_sep_7 , n.relu_7  = conv_dw(n.relu_6      , 256, 256, 1, test=self.test)
        n.conv_dw_8 , n.conv_sep_8 , n.relu_8  = conv_dw(n.relu_7      , 256, 256, 1, test=self.test)
        n.conv_dw_9 , n.conv_sep_9 , n.relu_9  = conv_dw(n.relu_8      , 256, 256, 1, test=self.test)
        n.conv_dw_10, n.conv_sep_10, n.relu_10 = conv_dw(n.relu_9      , 256, 256, 1, test=self.test)
        n.conv_dw_11, n.conv_sep_11, n.relu_11 = conv_dw(n.relu_10     , 256, 256, 1, test=self.test)
        n.conv_dw_12, n.conv_sep_12, n.relu_12 = conv_dw(n.relu_11     , 256, 512, 2, test=self.test)
        n.conv_dw_13, n.conv_sep_13, n.relu_13 = conv_dw(n.relu_12     , 512, 512, 1, test=self.test)
        
        
        n.up1 = upsample(n.relu_13, 512)
        n.skip1 = skip(n.relu_11, 256, 512, 1)
        n.up1 = eltsum(n.up1,n.skip1)
        n.filt1_dw, n.filt1_sep, n.filt1 = conv_dw(n.up1, 512, 16, 1, False, test=self.test)
        
        n.up2 = upsample(n.filt1, 16)
        n.skip2 = skip(n.relu_5, 128, 16, 1)
        n.up2 = eltsum(n.up2,n.skip2)
        n.filt2_dw, n.filt2_sep, n.filt2 = conv_dw(n.up2, 16, 16, 1, False, test=self.test)
        
        n.up3 = upsample(n.filt2, 16)
        n.skip3 = skip(n.relu_3, 64, 16, 1)
        n.up3 = eltsum(n.up3,n.skip3)
        n.filt3_dw, n.filt3_sep, n.filt3 = conv_dw(n.up3, 16, 16, 1, False, test=self.test)
        
        n.up4 = upsample(n.filt3, 16)
        n.skip4 = skip(n.relu_1, 32, 16, 1)
        n.up4 = eltsum(n.up4,n.skip4)
        n.filt4_dw, n.filt4_sep, n.filt4 = conv_dw(n.up4, 16, 16, 1, False, test=self.test)
        
        n.up5 = upsample(n.filt4, 16)
        n.filt5_dw, n.filt5_sep, n.filt5 = conv_dw(n.up5, 16, 16, 1, False, test=self.test)
        
        n.output_dw, n.output_sep, n.output = conv_dw(n.filt5, 16, 2, 1, False, test=self.test)
        
        n.loss = L.SoftmaxWithLoss(n.output, n.label,
                softmax_param=dict(axis=1, engine=1),
                loss_param=dict(normalization=1))
                
                # enum NormalizationMode {
                #     // Divide by the number of examples in the batch times spatial dimensions.
                #     // Outputs that receive the ignore label will NOT be ignored in computing
                #     // the normalization factor.
                #     FULL = 0;
                #     // Divide by the total number of output locations that do not take the
                #     // ignore_label.  If ignore_label is not set, this behaves like FULL.
                #     VALID = 1;
                #     // Divide by the batch size.
                #     BATCH_SIZE = 2;
                #     // Do not normalize the loss.
                #     NONE = 3;
                #   }
        
        return n.to_proto()

def make_net():
    with open('trainval.prototxt', 'w') as f:
        f.write(str(MobileNetHair('trainval').forward()))
        print 'trainval.prottxt updated'

    with open('test.prototxt', 'w') as f:
        f.write(str(MobileNetHair('test').forward()))
        print 'text.prottxt updated'
        
    # from train2deploy import train2deploy
    # deploy_prototxt_path = 'deploy.prototxt'
    # train2deploy('trainval.prototxt', (1, 3, 128, 128), deploy_prototxt_path)

if __name__ == '__main__':
    make_net()

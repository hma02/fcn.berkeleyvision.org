import caffe
import surgery, score

import numpy as np
import os
import sys

# try:
#     import setproctitle
#     setproctitle.setproctitle(os.path.basename(os.getcwd()))
# except:
#     pass



# init
caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

solver = caffe.AdamSolver('solver.prototxt')

# weights = '../hair-mn/siftflow-fcn8s-heavy.caffemodel'
# solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'upsample' in k]

surgery.interp(solver.net, interp_layers)

# scoring
# test = np.loadtxt('../data/hair/test.txt', dtype=str)
import glob
test = sorted( [s.split('/')[-1].split('.jpg')[0] for s in glob.glob('../data/hair/realdata/test/*.jpg')] )

for ind in range(100): # epoch=100
    solver.step(4500/4) # every epoch has iterations = 4500 images/ batchsize4
    # score manually from the histogram instead for proper evaluation
    score.seg_tests(solver, './test_img' , test, layer='output_sep', gt='label')

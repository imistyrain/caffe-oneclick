import os
import sys
sys.path.append(r'../../python')

MEAN_BIN = "../modeldef/mean.binaryproto"
MEAN_NPY = "../modeldef/mean.npy"

from caffe.proto import caffe_pb2
from caffe.io import blobproto_to_array

print('generating mean file...')

mean_blob = caffe_pb2.BlobProto()
mean_blob.ParseFromString(open(MEAN_BIN, 'rb').read())

import numpy as np
mean_npy = blobproto_to_array(mean_blob)
mean_npy_shape = mean_npy.shape
mean_npy = mean_npy.reshape(mean_npy_shape[1], mean_npy_shape[2], mean_npy_shape[3])

np.save(file(MEAN_NPY, 'wb'), mean_npy)

print('done...')
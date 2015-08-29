import caffe
from matplotlib import pyplot
import numpy as np

MODEL_FILE='../models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED='../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
IMAGE_FILE='../examples/images/cat.jpg'

caffe.set_mode_gpu()
mean = np.load('caffe/imagenet/ilsvrc_2012_mean.npy')
net=caffe.Classifier(MODEL_FILE,PRETRAINED,[256,256],mean,1.0,255.0,[2,1,0])
inputs = [caffe.io.load_image(IMAGE_FILE)]
predictions=net.predict(inputs )
imagecatory= predictions[0,:].argmax()
f=open("../data/ilsvrc12/synset_words.txt")
words=f.readlines()
f.close()

print 'prediction shape:', predictions.shape 
print predictions[0,:].argmax()
print predictions[0,:].max()
print words[imagecatory]

pyplot.plot(predictions[0,:])
pyplot.show()
import os
import sys
sys.path.insert(0, '../python')
import caffe
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2
import lmdb
from tqdm import tqdm
import logging
import cv2
import numpy as np

root_folder="data/"
batch_size = 64
test_batch_size = 100
input_size = [20,20]
G4 = 4*1024*1024*1024

def remove_if_exists(db):
  if os.path.exists(db):
    logger.info('remove %s'%db)
    shutil.rmtree(db)

def get_test_num(valpath = "util/val.txt"):
    with open(valpath) as f:
        lines = f.readlines()
        return len(lines)

def make_datum(img,label):
    return caffe_pb2.Datum(channels=3,width=input_size[0],height=input_size[1],label=label,
    data=np.rollaxis(img,2).tobytes())

def gen_data_layer(phase="train",uselmdb=True):
    if uselmdb:
        source = "lmdb/"+phase+"_lmdb"
        if not os.path.exists(source):
            print("creating "+source)
            os.makedirs(source)
            db = lmdb.open(source, map_size=G4)
            txn = db.begin(write=True)
            txtfile="util/"+phase+".txt"
            with open(txtfile) as f:
                lines = f.readlines()
                for i,line in tqdm(enumerate(lines)):
                    items = line.split()
                    imgpath = root_folder+"/"+items[0]
                    img = cv2.imread(imgpath)
                    if img is None:
                        logging.info("cannot read"+imgpath)
                    key = "%08d_data"%(i)
                    label=int(items[1])
                    txn.put(key,make_datum(img,label).SerializeToString())
                    if i %1000 == 0:
                        txn.commit()
                        txn = db.begin(write=True)
            db.close()
        data, label = L.Data(batch_size=batch_size, backend=P.Data.LMDB,source=source,transform_param=dict(scale=1./255), ntop=2)
    else:
        txtfile="util/"+phase+".txt"
        data, label = L.ImageData(image_data_param=dict(source=txtfile,root_folder=root_folder,batch_size=batch_size,shuffle=phase=="train",new_width=20,new_height=20),ntop=2,transform_param=dict(scale=1./255))
    return data,label

def lenet(phase="train",batch_size=64):
    n = caffe.NetSpec() 
    n.data, n.label = gen_data_layer(phase)
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.fc1 =   L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.fc1, in_place=True)
    n.fc2 = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.acc = L.Accuracy(n.fc2, n.label)
    n.loss =  L.SoftmaxWithLoss(n.fc2, n.label)
    return n

def lenet_deploy(net,deploy_net_file="util/deploy.prototxt"):
    deploy_net = net
    with open(deploy_net_file, 'w') as f:
        net_param = deploy_net.to_proto()
        del net_param.layer[0]
        del net_param.layer[-1]
        del net_param.layer[-1]
        net_param.name = 'lenet'
        net_param.input.extend(['data'])
        net_param.input_shape.extend([
            caffe_pb2.BlobShape(dim=[1, 3, input_size[0], input_size[1]])]) 
        f.write(str(net_param))

def gen_solver_txt(train_net_path, test_net_path):
    s = caffe_pb2.SolverParameter()
    s.train_net = train_net_path
    s.test_net.append(test_net_path)
    s.test_interval = 500
    s.test_iter.append(int(get_test_num()/test_batch_size))
    s.max_iter = 10000
    s.base_lr = 0.01
    s.lr_policy = 'step'
    s.gamma = 0.1
    s.power = 0.75
    s.stepsize = 5000
    s.momentum = 0.9
    s.weight_decay = 5e-4
    s.display = 1000
    s.snapshot = 5000
    s.snapshot_prefix = 'output/plate'
    s.solver_mode = caffe_pb2.SolverParameter.GPU
    return s

def main():
    train_net_path = 'util/train.prototxt'
    net = lenet('train',batch_size)
    with open(train_net_path, 'w') as f:
        f.write(str(net.to_proto()))
    test_net_path = 'util/test.prototxt'
    net = lenet('val',test_batch_size)
    with open(test_net_path, 'w') as f:
        f.write(str(net.to_proto()))
    lenet_deploy(net)
    solver_path = 'util/solver.prototxt'
    with open(solver_path, 'w') as f:
        f.write(str(gen_solver_txt(train_net_path, test_net_path)))
    caffe.set_mode_gpu()
    solver = caffe.get_solver(solver_path)
    solver.solve()

if __name__=="__main__":
    main()
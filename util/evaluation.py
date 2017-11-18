import os,argparse,sys,time,shutil
import numpy as np
import sys
sys.path.append('../python')
import caffe
import logging
from tqdm import tqdm

def create_logger(logdir="output"):
    time_str = time.strftime('%Y%m%d-%H%M%S')
    log_file = '{}/{}.log'.format(logdir, time_str)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(log_file),format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

def clearlasterrors(args):
    if os.path.exists("error"):
        subdirs=os.listdir(args.errordir)
        for subdir in subdirs:
            files=os.listdir(args.errordir+"/"+subdir)
            for file in files:
                os.remove(args.errordir+"/"+subdir+"/"+file)
            os.rmdir(args.errordir+"/"+subdir)

def loadmean(meanprotopath):
    blob = caffe.proto.caffe_pb2.BlobProto()
    blob.ParseFromString(open(meanprotopath, 'rb').read())   
    return np.array(caffe.io.blobproto_to_array(blob))[0]

def getclassifier(args):
    classifier = caffe.Classifier(args.modeldef, args.weights,image_dims=args.image_dims
    )
    ##mean=loadmean(args.meanfile).mean(1).mean(1),#raw_scale=255,channel_swap=[2,1,0]
    caffe.set_mode_gpu()
    return classifier

class EvalStatic:
    total = 0
    error = 0
    def __str__(self):
        return str(self.error)+","+str(self.total)+","+str(self.error*1.0/self.total)

def evaluationonebyone(args):
    labels=[w.split()[1] for w in open(args.labelfile).readlines()]
    classifier = getclassifier(args)
    start = time.time()
    if not os.path.exists(args.errordir):
        os.mkdir(args.errordir)
    subdirs=os.listdir(args.datadir)
    evalstatics=[]
    for subdir in subdirs:
        print(subdir+":")
        evalstatic=EvalStatic()
        files=os.listdir(args.datadir+'/'+subdir)
        evalstatic.total=len(files)
        for file in tqdm(files):
            imgpath=args.datadir+'/'+subdir+'/'+file
            inputs = [caffe.io.load_image(imgpath)]
            try:
                predictions = classifier.predict(inputs,oversample=False)
            except Exception as e:
                print(e)
            p=predictions[0,:].argmax()
            label=labels[p]
            if subdir!=label:
                logging.info(subdir+" "+file+":"+str(label))
                evalstatic.error=evalstatic.error+1
                if not os.path.exists(args.errordir+'/'+subdir):
                    os.mkdir(args.errordir+'/'+subdir)
                errorfilepath=args.errordir+'/'+subdir+'/'+file[:-4]+"_"+subdir+'_'+label+'.jpg'
                shutil.copy(imgpath,errorfilepath)
        evalstatics.append(evalstatic)
    logging.info("Done in %.2f s." % (time.time() - start))
    totalcount=0
    error=0
    for i,evalstatic in enumerate(evalstatics):
        error=error+evalstatic.error
        totalcount=totalcount+evalstatic.total
        logging.info(subdirs[i]+":"+str(evalstatic))
    logging.info("Toal error")
    logging.info(str(error)+" "+str(totalcount)+" "+str(error*1.0/totalcount))

def evaluation_batch(args):
    labels=[w.split()[1] for w in open(args.labelfile).readlines()]
    classifier=getclassifier(args)
    start = time.time()
    if not os.path.exists(args.errordir):
        os.mkdir(args.errordir)
    subdirs=os.listdir(args.datadir)
    evalstatics=[]
    for subdir in subdirs:
        print(subdir)
        evalstatic=EvalStatic()
        files=os.listdir(args.datadir+'/'+subdir)
        evalstatic.total=len(files)
        inputs=[caffe.io.load_image(args.datadir+'/'+subdir+'/'+file) for file in files]
        try:
            predictions = classifier.predict(inputs,oversample=False)
        except Exception as e:
            print(e)
        for i in tqdm(range(len(files))):
            p=predictions[i,:].argmax()
            label=labels[p]
            if subdir!=label:
                logging.info(subdir+" "+files[i]+":"+str(label))
                evalstatic.error=evalstatic.error+1
                if not os.path.exists(args.errordir+'/'+subdir):
                    os.mkdir(args.errordir+'/'+subdir)
                imgpath=args.datadir+"/"+subdir+"/"+files[i]
                errorfilepath=args.errordir+'/'+subdir+'/'+files[i][:-4]+"_"+subdir+'_'+label+'.jpg'
                shutil.copy(imgpath,errorfilepath)
        evalstatics.append(evalstatic)
    logging.info("Done in %.2f s." % (time.time() - start))
    totalcount=0
    error=0
    for i,evalstatic in enumerate(evalstatics):
        error=error+evalstatic.error
        totalcount=totalcount+evalstatic.total
        logging.info(subdirs[i]+":"+str(evalstatic))
    logging.info("Toal error")
    logging.info(str(error)+" "+str(totalcount)+" "+str(error*1.0/totalcount))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter",default=10000,help="caffemodel iter to evaluation")
    parser.add_argument("--datadir",default="data",help="datadir")
    parser.add_argument("--image_dims",default=[20,20],help="image_dims")
    parser.add_argument("--modeldef",default="util/deploy.prototxt",help="deploy file")
    parser.add_argument("--weights",default="models/plate999.caffemodel",help="caffemodel")
    parser.add_argument("--labelfile",default="models/labels.txt",help="label file")
    parser.add_argument("--meanfile",default="models/mean.binaryproto",help="meanfile")
    parser.add_argument("--errordir",default="error",help="errordir")
    parser.add_argument("--logfile",default="error.txt",help="log txt")
    parser.add_argument("--evaluationonebyone",default=True,help="log txt")
    parser.add_argument("--imgpath",default="data/0/0.jpg",help="image path")
    args = parser.parse_args()
    return args

def classification():
    args = get_args()
    args = parser.parse_args()
    labels=[w.split()[1] for w in open(args.labelfile).readlines()]
    classifier=getclassifier(args)
    inputs=[caffe.io.load_image(args.imgpath)]
    predictions = classifier.predict(inputs,oversample=False)
    p=predictions[0,:].argmax()
    label=labels[p]
    print(label,predictions[0,p])
    top_inds = predictions[0,:].argsort()[::-1][:5]

def evaluation():
    args = get_args()
    clearlasterrors(args)
    create_logger()
    if args.evaluationonebyone:
        evaluationonebyone(args)
    else:
        evaluation_batch(args)

if __name__=='__main__':
    evaluation()
    #classification()

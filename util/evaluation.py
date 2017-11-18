import os,argparse,sys,time,shutil
import numpy as np
import sys
sys.path.append(r'../../python')
import caffe

def clearlasterrors(args):
    print "Clearing last errors"
    subdirs=os.listdir(args.errordir)
    for subdir in subdirs:
        print subdir
        files=os.listdir(args.errordir+"/"+subdir)
        for file in files:
            os.remove(args.errordir+"/"+subdir+"/"+file)
        os.rmdir(args.errordir+"/"+subdir)
def loadmean(meanprotopath):
    blob = caffe.proto.caffe_pb2.BlobProto()
    blob.ParseFromString(open(meanprotopath, 'rb').read())   
    return np.array(caffe.io.blobproto_to_array(blob))[0]

def getclassifier(args):
    channel_swap=[2,1,0]
    raw_scale=255
    pretrainedmodel=args.weights_prefix+str(args.iter)+".caffemodel"
    classifier = caffe.Classifier(args.modeldef, pretrainedmodel,image_dims=args.image_dims,
                                  #mean=loadmean(args.meanfile).mean(1).mean(1),
                                  #raw_scale=raw_scale,
                                  channel_swap=channel_swap
                                  )
    caffe.set_mode_gpu()
    return classifier

class EvalStatic:
    total=0
    error=0
    def __str__(self):
        return str(self.error)+","+str(self.total)+","+str(self.error*1.0/self.total)

def evaluationonebyone(args):
    labels=[w.split()[1] for w in open(args.labelfile).readlines()]
    classifier=getclassifier(args)
    flog=open(args.logfile,'w')
    start = time.time()
    if not os.path.exists(args.errordir):
        os.mkdir(args.errordir)
    subdirs=os.listdir(args.datadir)
    evalstatics=[]
    for subdir in subdirs:
        print subdir+":"
        evalstatic=EvalStatic()
        files=os.listdir(args.datadir+'/'+subdir)
        evalstatic.total=len(files)
        for file in files:
            imgpath=args.datadir+'/'+subdir+'/'+file
            inputs = [caffe.io.load_image(imgpath)]
            try:
                predictions = classifier.predict(inputs,oversample=False)
            except Exception as e:
                print e
            p=predictions[0,:].argmax()
            label=labels[p]
            if subdir!=label:
                print subdir,file,label
                flog.write(str(subdir+'/'+str(file)+':'+str(label)+'\n'))
                evalstatic.error=evalstatic.error+1
                if not os.path.exists(args.errordir+'/'+subdir):
                    os.mkdir(args.errordir+'/'+subdir)
                errorfilepath=args.errordir+'/'+subdir+'/'+file[:-4]+"_"+subdir+'_'+label+'.jpg'
                shutil.copy(imgpath,errorfilepath)
        evalstatics.append(evalstatic)
    flog.write("Done in %.2f s.\n" % (time.time() - start))
    print("Done in %.2f s." % (time.time() - start))
    totalcount=0
    error=0
    for i,evalstatic in  enumerate(evalstatics):
        error=error+evalstatic.error
        totalcount=totalcount+evalstatic.total
        print subdirs[i],evalstatic
        flog.write(subdirs[i]+" "+str(evalstatic)+"\n")
    print error,totalcount,error*1.0/totalcount
    flog.write(str(error)+" "+str(totalcount)+" "+str(error*1.0/totalcount))
    flog.close()

def evaluation(args):
    labels=[w.split()[1] for w in open(args.labelfile).readlines()]
    classifier=getclassifier(args)
    start = time.time()
    if not os.path.exists(args.errordir):
        os.mkdir(args.errordir)
    subdirs=os.listdir(args.datadir)
    evalstatics=[]
    flog=open(args.logfile,'w')
    for subdir in subdirs:
        print subdir+":"
        evalstatic=EvalStatic()
        files=os.listdir(args.datadir+'/'+subdir)
        evalstatic.total=len(files)
        inputs=[caffe.io.load_image(args.datadir+'/'+subdir+'/'+file) for file in files]
        try:
            predictions = classifier.predict(inputs,oversample=False)
        except Exception as e:
            print e
        for i in range(len(files)):
            p=predictions[i,:].argmax()
            label=labels[p]
            if subdir!=label:
                print subdir,files[i],label
                flog.write(subdir,files[i],label,'\n')
                evalstatic.error=evalstatic.error+1
                if not os.path.exists(args.errordir+'/'+subdir):
                    os.mkdir(args.errordir+'/'+subdir)
                imgpath=args.datadir+"/"+subdir+"/"+files[i]
                errorfilepath=args.errordir+'/'+subdir+'/'+files[i][:-4]+"_"+subdir+'_'+label+'.jpg'
                shutil.copy(imgpath,errorfilepath)
        evalstatics.append(evalstatic)
    flog.write("Done in %.2f s." % (time.time() - start))
    print("Done in %.2f s." % (time.time() - start))
    totalcount=0
    error=0
    for i,evalstatic in  enumerate(evalstatics):
        error=error+evalstatic.error
        totalcount=totalcount+evalstatic.total
        print subdirs[i],evalstatic
        flog.write(subdirs[i]+str(evalstatic)+"\n")
    print error,totalcount,error*1.0/totalcount
    flog.write(str(error)+str(totalcount)+str(error*1.0/totalcount))
    flog.close()

def evaluationdir():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter",default=10000,help="caffemodel iter to evaluation")
    parser.add_argument("--datadir",default="../data",help="caffemodel iter to evaluation")
    parser.add_argument("--image_dims",default=[20,20],help="image_dims")
    parser.add_argument("--modeldef",default="../modeldef/deploy.prototxt",help="deploy file")
    parser.add_argument("--weights_prefix",default="../trainedmodels/platerecognition_iter_",help="caffemodel prefix")
    parser.add_argument("--labelfile",default="../modeldef/labels.txt",help="deploy file")
    parser.add_argument("--meanfile",default="../modeldef/mean.binaryproto",help="meanfile")
    parser.add_argument("--errordir",default="../error",help="errordir")
    parser.add_argument("--logfile",default="../log.txt",help="log txt")
    parser.add_argument("--evaluationonebyone",default=True,help="log txt")
    
    args = parser.parse_args()
    clearlasterrors(args)
    if args.evaluationonebyone:
        evaluationonebyone(args)
    else:
        evaluation(args)

def evaluation1image():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter",default=3000,help="caffemodel iter to evaluation")
    parser.add_argument("--image_dims",default=[20,20],help="image_dims")
    parser.add_argument("--modeldef",default="../modeldef/deploy.prototxt",help="deploy file")
    parser.add_argument("--weights_prefix",default="../trainedmodels/platerecognition_iter_",help="caffemodel prefix")
    parser.add_argument("--labelfile",default="../modeldef/labels.txt",help="deploy file")
    parser.add_argument("--meanfile",default="../modeldef/mean.binaryproto",help="meanfile")
    parser.add_argument("--logfile",default="../log.txt",help="log txt")
    parser.add_argument("--imgpath",default="../data/mask/299a24714_0.png",help="image path")
    args = parser.parse_args()
    labels=[w.split()[1] for w in open(args.labelfile).readlines()]
    classifier=getclassifier(args)
    inputs=[caffe.io.load_image(args.imgpath)]
    predictions = classifier.predict(inputs,oversample=False)
    p=predictions[0,:].argmax()
    label=labels[p]
    print label,predictions[0,p]
    top_inds = predictions[0,:].argsort()[::-1][:5]

if __name__=='__main__':
#        evaluation1image()
    evaluationdir()
import os,argparse,random,shutil

def conver2dir(args):
    aligneddir="E:/Palm/Datasets/CASIA-PalmprintV1Aligned"
    datadir=args.rootdir+"/"+args.dataname
    subdirs=os.listdir(aligneddir)
    for sub in subdirs:
        subdir=aligneddir+"/"+sub
        files=os.listdir(subdir)
        if len(file)==0:
            continue
        dstdir=datadir+"/"+sub
        if not os.path.exists(dstdir):
            os.mkdir(dstdir)
        for file in files:
            filepath=subdir+"/"+file
            dstpath=datadir+"/"+sub+"/"+file
            if file[7]=="l":
                shutil.copy(filepath,dstpath)

def main(args):
    datadir=args.rootdir+"/"+args.dataname
    print "loading data from "+datadir+":"
    trainfile=open("../util/train.txt","w");
    valfile=open("../util/val.txt","w");
    categoryfile=open("../modeldef/labels.txt",'w')
    paths=os.listdir(datadir)
    classindex=0
    trainpaths=[]
    valpaths=[]
    categorys=[]
    for subdir in paths:
        if(os.path.isdir(datadir+"/"+subdir)):
            categorys.append(str(classindex)+" "+subdir+"\n")
            files=os.listdir(datadir+"/"+subdir)
            files=[file for file in files]
            random.shuffle(files)
            print subdir,len(files)
            num2train=len(files)*args.trainrtaio
            for fileindex,file in enumerate(files):
                if fileindex<num2train:
                    trainpaths.append(subdir+"/"+file+" "+str(classindex)+"\n")#args.dataname+"/"+
                else:
                    valpaths.append(subdir+"/"+file+" "+str(classindex)+"\n")
            classindex=classindex+1

    for category in categorys:
        categoryfile.write(category)

    random.shuffle(trainpaths)
    random.shuffle(valpaths)
    print "writing to files...:"
    for trainpath in trainpaths:
        trainfile.write(trainpath)

    for valpath in valpaths:
        valfile.write(valpath)
    print(len(paths))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootdir",default="../",help="Directory of images to classify")
    parser.add_argument("--dataname",default="data",help="Dataset name")
    parser.add_argument("--trainrtaio",default=0.8,help="Train ratio ")
    parser.add_argument("--valrtaio",default=0.2,help="Val ratio")
    parser.add_argument("--testratio",default=0.1,help="Test ratoi")
    args = parser.parse_args()
#    conver2dir(args)
    main(args)
    print "Done"
import subprocess
import os

imgdir="../data/0001"
files=os.listdir(imgdir)
cmdprefix="../bin/classification "+"modeldef/deploy.prototxt "+"trainedmodels/plam_iter_5000.caffemodel "+"modeldef/mean.binaryproto "+"modeldef/labels.txt "
#files.sort(key= lambda x:int(x[:-4]))
for file in files:
    cmd=cmdprefix+imgdir+"/"+file
    subprocess.call(cmd)
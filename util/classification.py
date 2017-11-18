import numpy as np
import csv
import shutil
import os
trainsourcedir='train'
classfieddir='JDtest'
if not os.path.exists(classfieddir):
    os.mkdir(classfieddir)
predictedwriter= csv.writer(file('result.csv', 'wb'))
count=0
with open('train_label.txt','r') as trainfiletxt:
    processedfiles=open("processed.txt",'w')
    lines =trainfiletxt.readlines()
    for line in lines:
        trainfile =line.split(',')[0]
        label=line.split(',')[1][:-1]
        targetdir=os.path.join(classfieddir,label)
        if not os.path.exists(targetdir):
            os.mkdir(targetdir)
        shutil.copy(os.path.join(trainsourcedir,trainfile),os.path.join(targetdir,trainfile))
        count+=1
        processedfiles.write(str(count)+"\n")
        processedfiles.flush()
        #print count
        #predictedwriter.writerow([trainfile, label])
        #print line
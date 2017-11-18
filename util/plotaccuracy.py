#coding =utf-8
#迷若烟雨 @ G-wearable
import re
import matplotlib.pyplot as plt
import os
import sys
import datetime

def getlastesttraininfofilefromdir(logdir):
    logfiles=os.listdir(logdir)
    infologfiles=filter(lambda s:s.startswith('INFO'),logfiles)
    infologfiles=filter(lambda s:s.endswith('.txt'),infologfiles)
    if infologfiles:
        lastestfile=infologfiles[0]
        maxtm=0
        for logf in infologfiles:
            path = os.path.join(logdir, logf)  
            timestamp = os.path.getmtime(path)
            date = datetime.datetime.fromtimestamp(timestamp)
            if timestamp>maxtm:
                lastestfile=path
        return lastestfile
    else:
        return None
def plotaccuarcy():
    logdir='log'
    infologfile=getlastesttraininfofilefromdir(logdir)
    print infologfile
    if infologfile:
    #    infologfile='../log/INFO2015-11-19T19-45-15.txt'
        f=open(infologfile)
        lines=f.read()
        #print lines
        iterations=re.findall('Iteration \d*',lines)
        accuracysstrings=re.findall('accuracy = \d*.\d*',lines)
        trainlosstrings=re.findall('Train net output #0: loss = \d*.\d*',lines)
        testlossstrings=re.findall('Test net output #1: loss = \d*.\d*',lines)
        f.close()
        accuracys=[ac[11:] for ac in accuracysstrings]
        trainlosses=[loss[27:-1]for loss in trainlosstrings]
        testlosses=[loss[27:-1]for loss in testlossstrings]
        #for ac in accuracys:
        #    print ac
        plt.plot(range(len(accuracys)),accuracys)
        #plt.plot(range(len(trainlosses)),trainlosses)
        #plt.plot(range(len(testlosses)),testlosses)
        plt.show()
if __name__=="__main__":
    plotaccuarcy()
    raw_input()
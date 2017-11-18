import os
def renamefiles(datadir):
    paths=os.listdir(datadir)
    for subdir in paths:
        print subdir 
        files=os.listdir(datadir+"/"+subdir)
        fileindex=0
        for file in files:
            os.rename(datadir+"/"+subdir+"/"+file,datadir+"/"+subdir+"/"+str(fileindex)+".jpg")
            fileindex+=1
if __name__=="__main__":
    renamefiles("chars2")

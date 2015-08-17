#-*- coding:utf8 -*-#
"""
---------------------------------------
*功能：遍历data/user/里的每一个用户记录文件，然后读取并提用户的特征。
*用户-特征： {uid:[b1,b2,b3,b4]}  以用户uid为key，value为四种行为特征，31天累加(除了12-18那天)。
*保存：在data/dictionary/下生成uidfeature.pkl文件
---------------------------------------

"""
import os
import csv
import cPickle

def genUidFeature():
	dictionary = {}
	direction = "../data/user/"
	file_list = os.listdir(direction)

	#遍历每个用户文件
	for file_name in file_list:
	    file_path = direction+file_name
	    rows = csv.reader(open(file_path,'rb'))
	    rows.next()
	    for row in rows:
                
                if row[0]!="2014-12-18":
		    uid = file_name.split('.')[0]
		    if not dictionary.has_key(uid):
		        dictionary[uid]=[0,0,0,0]
		    dictionary[uid][int(row[2])-1] += 1


	f = open("../data/dictionary/uidfeature.pkl",'wb')
	cPickle.dump(dictionary,f,-1)
	f.close()





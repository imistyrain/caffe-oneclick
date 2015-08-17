#-*- coding:utf8 -*-#
"""
---------------------------------------
*功能：遍历/data/date里的文件，然后读取并提样本和特征。如"2014-12-18.csv"，文件里面出现的所有用户，以及用户有过行为的商品，都会分别生成样本(uid,iid)。
*举例：用户u1点击过i1、i2两件商品，买过i3这件商品，则该用户可以构建三个样本：(u1,i1)、（u1，i2）、（u1，i3）。
*样本-特征：样本名(uid,iid)以元组格式作为字典key，样本的特征向量以list格式[feat1,feat2...] 作为字典value。PS：字典的查找会快一点，以空间换时间。
*保存：生成的 样本-特征 保存为 "文件名.pkl" 文件，利用cPickle保存。
---------------------------------------

"""

import os
import csv
import cPickle

def genUidIid():
	os.mkdir("../data/dictionary/date")
	direction = "../data/date/"
	file_list = os.listdir(direction)
	for file_name in file_list:
	    file_path = direction+file_name
	    f = open(file_path,'rb')
	    rows = csv.reader(f)
	    rows.next()
	    dictionary = {} #{(uid,iid):[[b1,b2,b3,b4],[g1,g2..],[c1,c2..],[h1,h2..]]}
	    for row in rows:
		sample = (row[0],row[1])  # Attention: tuple is hashable,but list is not hashable
		if dictionary.has_key(sample):
		    dictionary[sample][0][int(row[2])-1] += 1
		    if dictionary[sample][1].count(row[3])==0:
		        dictionary[sample][1].append(row[3])
		    if dictionary[sample][2].count(row[4])==0:
		        dictionary[sample][2].append(row[4])
		    dictionary[sample][3].append(int(row[5]))
		else:
		    dictionary[sample]=[[0,0,0,0],[row[3]],[row[4]],[int(row[5])]]
		    dictionary[sample][0][int(row[2])-1] = 1
		dictionary[sample][3].sort()
	    f.close()
            
	    f = open("../data/dictionary/date/"+file_name.split('.')[0]+".pkl",'wb')
	    cPickle.dump(dictionary,f,-1)
	    f.close()


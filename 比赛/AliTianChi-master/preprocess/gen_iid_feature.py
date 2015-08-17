#-*- coding:utf8 -*-#
"""
---------------------------------------
*功能：遍历data/date里的每一个文件（按天分的），然后读取并提商品子集里的商品的特征。
*商品-特征： {iid:[b1,b2,b3,b4]}  以子集商品iid为key，b1~b4表示该商品被点击、收藏、加购物车、购买的次数，31天累加。
*保存：在data/dictionary/下生成iidfeature.pkl
---------------------------------------

"""
import os
import csv
import cPickle
import time


def genIidFeature():
	dictionary = {} #{iid:[b1,b2,b3,b4]}
	direction = "../data/date/"
	item = cPickle.load(open("../data/dictionary/item.pkl","rb"))
	file_list = os.listdir(direction)
	file_list.sort()
	for file_name in file_list:
	    #print "...process file:",file_name
	    file_path = direction+file_name
	    f = open(file_path,'rb')
	    rows = csv.reader(f)
	    rows.next()
	    for row in rows:
		iid = row[1]
		if item.has_key(iid):
		    if not dictionary.has_key(iid):
		        dictionary[iid]=[0,0,0,0]
		    dictionary[iid][int(row[2])-1] += 1           
	    f.close()

	f = open("../data/dictionary/iidfeature.pkl",'wb')
	cPickle.dump(dictionary,f,-1)
	f.close()





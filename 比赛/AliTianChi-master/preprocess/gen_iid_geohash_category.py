#-*- coding:utf8 -*-#
"""
---------------------------------------
*功能：将tianchi_mobile_recommend_train_item.csv保存为字典（哈希表），方便查找。
*格式：以商品iid为key，对应商品位置和类别表[geohash,category]为value

---------------------------------------

"""

import os
import csv
import cPickle

def genIid():
    os.mkdir("../data/dictionary")
    file_path = "../data/tianchi_mobile_recommend_train_item.csv"

    f = open(file_path,'rb')
    rows = csv.reader(f)
    rows.next()
    dictionary = {} 
    for row in rows:
        dictionary[row[0]] = [row[1],row[2]]
    f = open("../data/dictionary/item.pkl",'wb')
    cPickle.dump(dictionary,f,-1)
    f.close()




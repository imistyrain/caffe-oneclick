#-*-coding:utf-8-*-

"""
遍历'data/date/'目录下的所有csv文件，按照用户分割，生成'data/user'目录以及用户文件.

用户文件内容格式：

'date','item_id','behavior_type','user_geohash','item_category','hour'

"""

import os
import csv
import time

#记录已存在的user_id.csv
user_dictionary = {}



def writeByUser(user_id,words):
    file_name = user_id+".csv"
    os.chdir("../data/user/")
    if not user_dictionary.has_key(user_id):
        user_dictionary[user_id] = True
        f = open(file_name,'a')
        write = csv.writer(f)
        write.writerow(['date','item_id','behavior_type','user_geohash','item_category','hour'])
        write.writerow(words)
        f.close()
    else:
        f = open(file_name,'a')
        write = csv.writer(f)
        write.writerow(words)
        f.close()
    os.chdir("../../preprocess/")


def splitByUser():
    os.mkdir("../data/user/")
    directory = "../data/date/"
    csvlist = os.listdir(directory)
    csvlist.sort()
    for eachcsv in csvlist:
        f = open(directory+eachcsv)
        rows = csv.reader(f)
        rows.next()
        for row in rows:
            user_id = row[0]
            words = [eachcsv.split('.')[0]]
            words.extend(row[1:])
            writeByUser(user_id,words)

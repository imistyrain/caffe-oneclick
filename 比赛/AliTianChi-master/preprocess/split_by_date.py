#-*-coding:utf-8-*-
"""
将tianchi_mobile_recommend_train_user.csv按照日期分割为31份**.csv文件，放在'/data/date/'目录下。
生成的**.csv文件内容格式如下：

user_id, item_id, behavior_type,user_geohash,item_category,    hour
99512554,37320317,  3,            94gn6nd,    9232,             20

"""

import csv
import os


#记录已存在的date.csv
date_dictionary = {}
   
#将words写入date.csv文件最后一行，文件打开采用'a'模式，即在原文件后添加（add）
def writeByDate(date,words):
    file_name = date+".csv"
    os.chdir('../data/date/')
    if not date_dictionary.has_key(date):
        date_dictionary[date] = True
        f = open(file_name,'a')
        write = csv.writer(f)
        write.writerow(['user_id','item_id','behavior_type','user_geohash','item_category','hour'])
        write.writerow(words)
        f.close()
    else:
        f = open(file_name,'a')
        write = csv.writer(f)
        write.writerow(words)
        f.close()
    os.chdir('../../preprocess/')


#主函数
def splitByDate():
    os.mkdir('../data/date')
    f = open("../data/tianchi_mobile_recommend_train_user.csv")
    rows = csv.reader(f)
    rows.next()
    for row in rows:
        date = row[-1].split(" ")[0]
        hour = row[-1].split(" ")[1]
        words = row[0:-1]
        words.append(hour)
        writeByDate(date,words)

#-*-coding:utf-8-*-
"""
1.运行split_by_date.py，在data目录下生成date文件夹以及文件
2.运行split_by_user.py，在data目录下生成user文件夹以及文件
3.运行gen_iid_geohash_category.py，在data目录下生成dictionary文件夹以及item.pkl文件
4.运行gen_uid_iid.py，在data/dictionary目录下生成date文件夹以及*.pkl文件
5.运行gen_iid_feature.py，在data/dictionary目录下生成iidfeature.pkl文件
6.运行gen_uid_feature.py，在data/dictionary目录下生成uidfeature.pkl文件

"""

import time
from split_by_date import splitByDate
from split_by_user import splitByUser
from gen_iid_geohash_category import genIid
from gen_uid_iid import genUidIid
from gen_iid_feature import genIidFeature
from gen_uid_feature import genUidFeature


if __name__ == "__main__":
    print "====================================="
    t0 = time.time()
    splitByDate()
    t1 = time.time()
    print "It takes %f s to split by date,generate 'data/date/*.csv'" %(t1-t0)
    splitByUser()
    t2 = time.time()
    print "It takes %f s to split by user,generate 'data/user/*.csv'" %(t2-t1)
    genIid()
    t3 = time.time()
    print "It takes %f s to make dictionary{iid:[geohash,category]},generate 'data/dictionary/item.pkl'" %(t3-t2)
    genUidIid()
    t4 = time.time()
    print "It takes %f s to make dictionary{(uid,iid):[[b1,b2,b3,b4],[g1,g2..],[c1,c2..],[h1,h2..]]},generate 'data/dictionary/date/*.pkl'" %(t4-t3)
    genIidFeature()
    t5 = time.time()
    print "It takes %f s to make dictionary{iid:[b1,b2,b3,b4]},generate 'data/dictionary/iidfeature.pkl'" %(t5-t4)
    genUidFeature()
    t6 = time.time()
    print "It takes %f s to make dictionary{uid:[b1,b2,b3,b4]},generate 'data/dictionary/uidfeature.pkl'" %(t6-t5)
    print "====================================="

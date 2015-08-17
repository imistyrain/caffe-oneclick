#-*-coding:utf-8-*-#
"""
根据以下规则生成提交文件,F1达到10+%（切换后的数据）
(1) 12月18号15点后加购物车且当天没买(根据历史统计信息得出的规律)
(2) 加入12点、13点加购物车的(试出来的)
(3) 过滤掉17048049、24029586、128289511这三个用户（这三个用户在18号当天加购物车量特别多，且从历史信息得出他们转购率很低）
"""

import cPickle
import csv


#存储 (uid,iid)
result = {}


item = cPickle.load(open("../data/dictionary/item.pkl","rb"))
day = cPickle.load(open("../data/dictionary/date/2014-12-18.pkl","rb"))

for key in day:
    uid,iid = key
    
    if  uid!= "17048049" and uid!="24029586" and uid!= "128289511" and item.has_key(iid) and day[key][0][2]>0 and day[key][0][3]==0:
        rows = csv.reader(open("../data/user/"+key[0]+".csv","rb"))
        rows.next()           
        for row in rows:
            if row[0] == "2014-12-18" and row[1] == key[1] and row[2] == "3" and (int(row[-1])>=15 or int(row[-1])==12 or int(row[-1])==13):
                result[key] = 1


#写入文件
f = open("tianchi_mobile_recommendation_predict.csv","wb")
write = csv.writer(f)
write.writerow(["user_id","item_id"])
total = 0
for key in result:
    write.writerow(key)
    total += 1 
print "generate submission file,total %d  (uid,iid)" %total
f.close()

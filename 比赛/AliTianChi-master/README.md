2015阿里巴巴天池大数据竞赛-阿里移动推荐
===

**竞赛介绍：**[链接](http://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100066.333.2.umhl4N&raceId=1)

目录说明
---
- **data**  *存放数据*
- **preprocess**    *数据预处理*
- **rule**   *根据规则生成提交文件*
- **model**   *训练机器学习模型*
- **statistics** *统计历史数据的信息*
	




代码使用说明
---

- fork本repo，非Github用户请点右下角的`Downlown ZIP`

- 解压后，将`tianchi_mobile_recommend_train_user.csv`以及`tianchi_mobile_recommend_train_item.csv`放入`/data/`目录下

- 仅需两个步骤即可获得一份提交文件

	- 第一步，进入`/preprocess/`目录，运行`data_preprocess.py`
	- 第二步，进入`/rule/`目录，运行`gen_submission_by_rule.py`


- 完成上面两个步骤后，在`/rule/`目录下会生成一份`tianchi_mobile_recommendation_predict.csv`文件，提交它。在切换数据前F1为7.6%，切换数据后的F1为8+%


如何获得更高的F1?
---
我们总结了一下，大概有三种方法：

1. 做好特征工程，选对模型，调好参。
2. 用规则硬爆，了解业务，从历史数据中发现规则。
3. 多用一些小号，多提交几次。我跟队友戏称，"给我10个小号，我也能上榜首"。当然，我们没刷小号，一天只提交两份（队伍三个人）。

4月1号比赛开始后，我和几个队友都是单干，各自试自己的模型，我队友 [pinfish](https://github.com/pinfish888) 习惯于用SQL，我则习惯于用Python和相应的包。在头两天，我基本上都是在做数据处理和特征提取，第三天训练了一个随机森林和逻辑回归，随机森林的效果比逻辑回归好不少，排到第四页。之后，尝试了一下规则，竟然上了首页，顿觉做模型是在浪费时间，如果是为了进第二赛季，用规则足够。在差不多4月7号后，我们队伍三个人基本上都各自有事，忙着科研，忙着写论文。直到切换数据的前几天，我每天抽出一两个小时，尝试各种规则。切换数据后，基本上也是用这些规则，稳稳地进了第二赛季。

`/rule/`目录下的`gen_submission_by_rule1.py`文件，便包含了一些不错的规则，记得F1得分是10.3%




补充说明
---

- **纯Python**，无任何依赖项。

- 关于代码实现的功能，在每份代码文件中均有注释，代码可能写得比较乱，也可能有bug，欢迎issues。
	
- **建议在Linux下运行**；在我的PC上（core-i7），上面两个步骤总共花了不到10分钟。


# AmazonSentimentAnalysis
基于Bi-LSTM的亚马逊评论情感二分类模型及可视化<br>
本项目对电商平台亚马逊上的商品评论情感二分类数据集进行筛选清洗等预处理后，通过基于Word2vec的Bi-LSTM算法建立情感二分类模型，并基于该模型最终实现一款可以对单条商品评论进行情感趋向判断，对来自本地的和来自网页的大量商品评论数据进行情感分类并得到分类比例的桌面级应用程序。<br>

数据地址 https://www.kaggle.com/datasets/nabamitachakraborty/amazon-reviews<br>
词向量地址 https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz (国内无法下载，可直接搜索百度云下载)

## 运行说明
1.下载数据及词向量
2.运行binTotxt.py转换词向量格式
3.运行main.py训练模型
4.运行guiActive.py显示可视化界面
注：可通过createTest.py选取部分数据进行训练测试，加快训练速度
## 界面展示
![image](https://user-images.githubusercontent.com/68805593/230900781-85e11b90-0bc6-4e50-ad46-20d947e9f1f3.png)<br>
![image](https://user-images.githubusercontent.com/68805593/230900946-535756be-083e-4b23-bbee-5222728a5c8b.png)<br>
![image](https://user-images.githubusercontent.com/68805593/230900993-9053f19c-c4e4-4e20-a9f1-b563b62b8a3a.png)<br>


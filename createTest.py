# #数据集生成
import pandas as pd
import numpy as np
import random

# 100000原始数据->40000条测试数据
file1 = pd.read_csv('1000000origin.csv')
scale = list(range(1,900000)) # 生成随机数范围，0 ~ file1 的长度
num = 40500
randoms = random.sample(scale, num)
data=file1.iloc[randoms]
data.to_csv('40000eval.csv',header=None,index=None)

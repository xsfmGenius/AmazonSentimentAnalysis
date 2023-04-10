# 二改：更换数据集，训练集360万条数据，测试集40万条数据


# #模型训练
#coding:utf-8
import warnings
import time
import spacy
import pandas as pd
from torchtext import data
from torchtext.vocab import Vectors
import os
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import pickle
from matplotlib import pyplot as plt

# #忽略警告
warnings.filterwarnings('ignore')
T0 = time.perf_counter()

# # 读入数据,规格化(2积极->1,1消极->0)     二改：冗余内容过多，列名读取保存等多此一举
# 测试集
testfile=pd.read_csv('test.csv',dtype=str,encoding='ISO-8859-1',header=None)
# testfile=pd.read_csv('testdata/test.csv',dtype=str,encoding='ISO-8859-1',header=None)
testfile = testfile.drop(0)
testCate=testfile[1].astype('category').cat.codes
print("测试集\n{}".format(testCate.value_counts()))
datalist=list(zip(testfile[0],testCate))
test=pd.DataFrame(data=datalist,columns=['review','label'])
test.to_csv('test_create.csv',header=None,index=None)
# 训练集
trainfile=pd.read_csv('train.csv',dtype=str,encoding='ISO-8859-1',header=None)
# trainfile=pd.read_csv('testdata/train.csv',dtype=str,encoding='ISO-8859-1',header=None)
trainfile = trainfile.drop(0)
trainCate=trainfile[1].astype('category').cat.codes
print("训练集\n{}".format(trainCate.value_counts()))
datalist=list(zip(trainfile[0],trainCate))
train=pd.DataFrame(data=datalist,columns=['review','label'])
train.to_csv('train_create.csv',header=None,index=None)
T1 = time.perf_counter()
print('规格化t1-t0={}'.format(T1-T0))
# print(trainCate)
# print(trainCate[15])
# print(train[0])
# print(len(train[0]))

# # 文本处理，加载数据
spacy_en = spacy.load('en_core_web_sm')
def tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text) if not tok.is_punct | tok.is_space| tok.is_stop]
# | tok.is_stop
#定义Field(一个能够加载、预处理和存储文本数据和标签的对象)
Review=data.Field(sequential=True,tokenize=tokenizer,lower=True)
Label=data.LabelField(sequential=False, use_vocab=False)
fields=[("rev",Review),("lab",Label)]
# 加载测试集数据  二改：可使用splits同时读取
testDataSet=data.TabularDataset(
    path='test_create.csv',
    format='csv',
    fields=fields,
    skip_header=False
)
# 加载训练集数据
trainDataSet=data.TabularDataset(
    path='train_create.csv',
    format='csv',
    fields=fields,
    skip_header=False
)
T2 = time.perf_counter()
print('加载数据t2-t1={}'.format(T2-T1))
# print(len(testDataSet))
# print(vars(testDataSet[0]))
#
# print(len(trainDataSet))
# print(vars(trainDataSet[0]))

# #建立词表
cache = '.vector_cache'
if not os.path.exists(cache):
    os.mkdir(cache)
vectors=Vectors(name='GoogleNews-vectors-negative300.txt',cache=cache)
Review.build_vocab(trainDataSet,vectors=vectors,max_size=300000)
Label.build_vocab((trainDataSet))
vocab=Review.vocab.vectors
print(len(vocab))
print(vocab.shape)
print(Review.vocab.freqs.most_common(10))
print(Review.vocab.itos[:10])
# print(Review.vocab.stoi)
# #保存词表
output=open('vocab.pkl','wb')
pickle.dump(Review.vocab,output)
output.close()

T3 = time.perf_counter()
print('建立词表t3-t2={}'.format(T3-T2))


# #构建迭代器
batch_size=100
device="cuda" if torch.cuda.is_available() else "cpu"
# 训练集
train_iter=data.BucketIterator(trainDataSet,
    batch_size=batch_size,
    device=device,
    sort_within_batch=True,
    sort_key=lambda x : len(x.rev)
)
#测试集
test_iter=data.Iterator(testDataSet,
    batch_size=batch_size,
    device=device,
    sort=False,
    sort_within_batch=False,
)

# #定义模型，双向LSTM
class LSTM(nn.Module):
    def __init__(self,embedding_dim,hidden_size,vocab_size,dropout_rate):
        super(LSTM,self).__init__()
        self.embedding=nn.Embedding(vocab_size,embedding_dim)
        self.encoder=nn.LSTM(input_size=embedding_dim,hidden_size=hidden_size,bidirectional=True,num_layers=2,dropout = dropout_rate)
        self.predictor=nn.Linear(hidden_size*2,2)
        self.dropout=nn.Dropout(dropout_rate)

    def forward(self,seq):
        output,(hidden,cell)=self.encoder(self.embedding(seq))
        hidden=torch.cat([hidden[-2],hidden[-1]],dim=1)
        hidden=self.dropout(hidden)
        preds=self.predictor(hidden)
        return preds

# len（vocab.size)
# #建模，嵌入词向量
lstm_model=LSTM(hidden_size=50,embedding_dim=300,vocab_size=300002,dropout_rate=0.5)
lstm_model.embedding.weight.data.copy_(vocab)
# lstm_model.embedding.weight.requires_grad=False
if hasattr(torch.cuda, 'empty_cache'):
	torch.cuda.empty_cache()
print(lstm_model)
lstm_model.to(device)
#weight_decay=0.00001
# #优化器，损失函数
optimizer=optim.SGD(lstm_model.parameters(),lr=0.005,weight_decay=0.00001)
criterion=nn.CrossEntropyLoss()

# #绘图准备
epochPlt=[]
trainLossPlt=[]
testLossPlt=[]
trainAccPlt=[]
testAccPlt=[]

# #训练测试函数
def train_val_test(model,optimizer,criterion,train_iter,test_iter,epochs):
    best_acc=0.0
    for epoch in range(1,epochs+1):
        print("Epoch:{}".format(epoch))
        epochPlt.append(epoch)
        # 训练
        time.sleep(1)
        train_loss=0.0
        train_corret = 0.0
        model.train()
        for indices,batch in tqdm(enumerate(train_iter),total =len(train_iter),leave = False,desc="train"):
            optimizer.zero_grad()
            context = batch.rev.to(device)
            target = batch.lab.to(device)
            outputs=model(context)
            loss=criterion(outputs,target)
            loss.backward()
            optimizer.step()
            train_loss+=loss.data.item()*batch.rev.size(1)
            preds = outputs.argmax(1)
            train_corret += preds.eq(target.view_as(preds)).sum().item()
        train_loss/=len(trainCate)
        trainLossPlt.append(train_loss)
        train_acc = train_corret / (len(trainCate))
        trainAccPlt.append(train_acc)
        print("Train Loss:{:.2f}".format(train_loss))
        print("Train_Accuracy:{:.2f}".format(100*train_acc))
        # 测试
        time.sleep(1)
        model.eval()
        test_corret=0.0
        test_loss=0.0
        with torch.no_grad():
            for idx,batch in tqdm(enumerate(test_iter),total =len(test_iter),leave = False,desc="test"):
                context=batch.rev.to(device)
                target=batch.lab.to(device)
                outputs=model(context)
                loss=criterion(outputs,target)
                test_loss+=loss.data.item()*context.size(1)
                preds=outputs.argmax(1)
                test_corret+=preds.eq(target.view_as(preds)).sum().item()
            test_loss/=len(testCate)
            testLossPlt.append(test_loss)
            test_acc=test_corret/(len(testCate))
            testAccPlt.append(test_acc)
            print("Test Loss:{:.2f}".format(test_loss))
            print("Test_Accuracy:{:.2f}".format(100*test_acc))
            # 绘图
            plt.plot(epochPlt, trainLossPlt)
            plt.plot(epochPlt, testLossPlt)
            plt.plot(epochPlt, trainAccPlt)
            plt.plot(epochPlt, testAccPlt)
            plt.ylabel('loss/Accuracy')
            plt.xlabel('epoch')
            plt.legend(['trainLoss', 'validationLoss', 'trainAcc', 'validationAcc'], loc='best')
            plt.show()
            # 保存
            if (test_acc > best_acc):
                best_acc = test_acc
                torch.save(lstm_model, 'Best-lstm.pt')
                print("模型已保存")


if hasattr(torch.cuda, 'empty_cache'):
	torch.cuda.empty_cache()
train_val_test(lstm_model,optimizer,criterion,train_iter,test_iter,epochs=100)
T4 = time.perf_counter()
print('模型训练t4-t3={}'.format(T4-T3))
# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
# https://blog.csdn.net/weixin_41744192/article/details/115270178
# https://blog.csdn.net/l_aiya/article/details/126412008
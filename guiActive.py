# #界面启动
#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from PyQt5.QtWidgets import QApplication,QMainWindow,QFileDialog,QAction, QLineEdit, QFormLayout, QHBoxLayout, QPushButton,QMessageBox
from PyQt5 import QtCore, QtGui, QtWidgets
import home # module test.py
from PyQt5.QtChart import QChart, QChartView, QPieSeries, QPieSlice
from PyQt5.QtGui import QPainter, QPen
import requests
import json
from googletrans import Translator
from spider import spider

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
import numpy as np
import pickle

# 忽略警告
warnings.filterwarnings('ignore')

# 分词函数
def tokenizer(text):  # create a tokenizer function
    return [tok.text for tok in spacy_en.tokenizer(text) if not tok.is_punct | tok.is_space]

# 模型结构
class LSTM(nn.Module):
    def __init__(self,embedding_dim,hidden_size,vocab_size,dropout_rate):
        super(LSTM,self).__init__()
        self.embedding=nn.Embedding(vocab_size,embedding_dim)
        self.encoder=nn.LSTM(input_size=embedding_dim,hidden_size=hidden_size,bidirectional=True,num_layers=2,dropout=dropout_rate)
        self.predictor=nn.Linear(hidden_size*2,2)
        self.dropout=nn.Dropout(dropout_rate)

    def forward(self,seq):
        output,(hidden,cell)=self.encoder(self.embedding(seq))
        hidden=torch.cat([hidden[-2],hidden[-1]],dim=1)
        hidden=self.dropout(hidden)
        preds=self.predictor(hidden)
        return preds

# 界面响应
class pages_window(home.Ui_Form,QMainWindow):
    def __init__(self):
        super(pages_window,self).__init__()
        self.setupUi(self)
        self.stackedWidget.setCurrentIndex(0)
        # 隐藏结果文本控件
        self.label_10.setHidden(True)
        self.label_11.setHidden(True)
        self.label_17.setHidden(True)
        self.label_30.setHidden(True)
        self.label_31.setHidden(True)
        self.label_32.setHidden(True)
        #页面转换
        self.pushButton_1.clicked.connect(self.displayPage1)
        self.pushButton_2.clicked.connect(self.displayPage2)
        self.pushButton_3.clicked.connect(self.displayPage3)
        #单条评论
        self.pushButton.clicked.connect(self.singleRev)
        #评论文件
        self.pushButton_4.clicked.connect(self.revFile)
        #商品网址
        self.pushButton_5.clicked.connect(self.revOnline)

    #页面转换
    def displayPage1(self):
        self.stackedWidget.setCurrentIndex(0)
        self.pushButton_1.setStyleSheet("background-color: rgb(238, 238, 238);")
        self.pushButton_2.setStyleSheet("background-color: rgb(210, 210, 210);")
        self.pushButton_3.setStyleSheet("background-color: rgb(210, 210, 210);")

    def displayPage2(self):
        self.stackedWidget.setCurrentIndex(1)
        self.pushButton_2.setStyleSheet("background-color: rgb(238, 238, 238);")
        self.pushButton_1.setStyleSheet("background-color: rgb(210, 210, 210);")
        self.pushButton_3.setStyleSheet("background-color: rgb(210, 210, 210);")

    def displayPage3(self):
        self.stackedWidget.setCurrentIndex(2)
        self.pushButton_3.setStyleSheet("background-color: rgb(238, 238, 238);")
        self.pushButton_2.setStyleSheet("background-color: rgb(210, 210, 210);")
        self.pushButton_1.setStyleSheet("background-color: rgb(210, 210, 210);")

    #单条评论
    def singleRev(self):
        try:
           str=self.textEdit.toPlainText()
           str = str.replace('\n', ' ')
           print(str);
           # 翻译
           translator = Translator(service_urls=[
               'translate.google.cn'
           ])
           translation = translator.translate(str, dest='zh-CN')
           self.textEdit_2.setText(translation.text)
           # 情感分析
           tokenized = [tok.text for tok in spacy_en.tokenizer(str) if not tok.is_punct | tok.is_space]
           index = [vocab.stoi[i] for i in tokenized]
           device = "cuda" if torch.cuda.is_available() else "cpu"
           preTensor = torch.LongTensor(index).to(device)
           preTensor = preTensor.unsqueeze(1)
           prediction = bestModel(preTensor)
           preds = prediction.argmax(1)
           if (preds == 0):
               self.radioButton.setChecked(False)
               self.radioButton_2.setChecked(True)
           else:
               self.radioButton_2.setChecked(False)
               self.radioButton.setChecked(True)
        except BaseException:
            msg_box = QMessageBox(QMessageBox.Critical, '错误', '请检查输入！')
            msg_box.exec_()


    #评论文件
    def revFile(self):
        #显示评论
        fileName=self.lineEdit.text()
        column=self.spinBox.value()
        try:
            originalFile = pd.read_csv(fileName, dtype=str, header=None, encoding='ISO-8859-1',skipinitialspace=True)
        except BaseException:
            msg_box2 = QMessageBox(QMessageBox.Critical, '错误', '请输入正确的文件路径！')
            msg_box2.exec_()
        else:
            originalFile = originalFile.drop(0)
            self.listWidget.clear()
            try:
                self.listWidget.addItems(originalFile[column])
            except BaseException:
                msg_box3 = QMessageBox(QMessageBox.Critical, '错误', '请输入正确的列号！')
                msg_box3.exec_()
            else:
                # 情感分析
                tempData = pd.DataFrame(data=originalFile[column])
                tempData.to_csv('E:\\Microsoft Visual Studio\\MyProjects\\emoAna\\tempdata.csv', header=None,
                                index=None)
                Review = data.Field(sequential=True, tokenize=tokenizer, lower=True)
                Label = data.LabelField(sequential=False, use_vocab=False)
                fields = [("rev", Review)]
                tempDataSet = data.TabularDataset(
                    path='tempdata.csv',
                    format='csv',
                    fields=fields,
                    skip_header=False
                )
                Review.vocab = vocab
                Label.build_vocab((tempDataSet))
                temp_iter = data.Iterator(tempDataSet,
                                          batch_size=batch_size,
                                          device=device,
                                          sort=False,
                                          sort_within_batch=False,
                                          )
                sum = 0
                with torch.no_grad():
                    for idx, batch in enumerate(temp_iter):
                        context = batch.rev.to(device)
                        outputs = bestModel(context)
                        preds = outputs.argmax(1)
                        num1 = preds.sum()
                        sum += num1
                # 绘制饼图
                self.series = QPieSeries()
                self.series.append("积极", sum.item())
                self.series.append("消极", len(tempDataSet) - sum.item())
                self.chart = QChart()
                self.chart.addSeries(self.series)
                self.chartview = QChartView(self.chart, self.page)
                self.chartview.setRenderHint(QPainter.Antialiasing)  # 抗锯齿
                self.chartview.setGeometry(QtCore.QRect(40, 235, 261, 256))
                self.chartview.show()
                # 显示数据
                self.label_10.setHidden(False)
                self.label_11.setHidden(False)
                self.label_17.setHidden(False)
                self.label_18.setText(str(len(tempDataSet)))
                posPer = 100 * sum.item() / len(tempDataSet)
                negPer = 100 * (len(tempDataSet) - sum.item()) / len(tempDataSet)
                self.label_12.setText("{}/{:.1f}%".format(sum.item(), posPer))
                self.label_13.setText("{}/{:.1f}%".format(len(tempDataSet) - sum.item(), negPer))
                os.remove("tempdata.csv")

    # 商品网址
    def revOnline(self):
        try:
            # 显示评论
            net = self.lineEdit_2.text()
            spider(net)
            originalFile = pd.read_csv("comments.csv", dtype=str, header=None, encoding='UTF-8', skipinitialspace=True)
            self.listWidget_2.clear()
            print(originalFile[0])
            self.listWidget_2.addItems(originalFile[0])
            # 情感分析
            Review = data.Field(sequential=True, tokenize=tokenizer, lower=True)
            fields = [("rev", Review)]
            commentDataSet = data.TabularDataset(
                path='comments.csv',
                format='csv',
                fields=fields,
                skip_header=False
            )
            Review.vocab = vocab
            com_iter = data.Iterator(commentDataSet,
                                     batch_size=batch_size,
                                     device=device,
                                     sort=False,
                                     sort_within_batch=False,
                                     )
            sum = 0
            with torch.no_grad():
                for idx, batch in enumerate(com_iter):
                    context = batch.rev.to(device)
                    outputs = bestModel(context)
                    preds = outputs.argmax(1)
                    num1 = preds.sum()
                    sum += num1
            # 绘制饼图
            self.series_2 = QPieSeries()
            self.series_2.append("积极", sum.item())
            self.series_2.append("消极", len(commentDataSet) - sum.item())
            self.chart_2 = QChart()
            self.chart_2.addSeries(self.series_2)
            self.chartview_2 = QChartView(self.chart_2, self.page_2)
            self.chartview_2.setRenderHint(QPainter.Antialiasing)  # 抗锯齿
            self.chartview_2.setGeometry(QtCore.QRect(40, 235, 261, 256))
            self.chartview_2.show()
            # 显示数据
            self.label_30.setHidden(False)
            self.label_31.setHidden(False)
            self.label_32.setHidden(False)
            posPer = 100 * sum.item() / len(commentDataSet)
            negPer = 100 * (len(commentDataSet) - sum.item()) / len(commentDataSet)
            self.label_33.setText(str(len(commentDataSet)))
            self.label_28.setText("{}/{:.1f}%".format(sum.item(), posPer))
            self.label_29.setText("{}/{:.1f}%".format(len(commentDataSet) - sum.item(), negPer))
        except BaseException:
            msg_box4 = QMessageBox(QMessageBox.Critical, '错误', '无法获取评论，请检查输入！')
            msg_box4.exec_()




if __name__ == '__main__':
    spacy_en = spacy.load('en_core_web_sm')
    vocab_file = open('vocab.pkl', 'rb')
    vocab = pickle.load(vocab_file)
    bestModel = torch.load("Best-lstm.pt")
    batch_size = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"

    app = QApplication(sys.argv)
    myMainWindow = pages_window()
    myMainWindow.show()
    sys.exit(app.exec_())
# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助

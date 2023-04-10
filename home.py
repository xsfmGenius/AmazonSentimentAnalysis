# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'home.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets

class mylineedit(QtWidgets.QLineEdit):
    def mouseDoubleClickEvent(self, e):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, '打开文件', '.', '*.csv')
        self.setText(fname)



class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(709, 626)
        Form.setMinimumSize(QtCore.QSize(709, 626))
        Form.setMaximumSize(QtCore.QSize(709, 626))
        self.layoutWidget = QtWidgets.QWidget(Form)
        self.layoutWidget.setGeometry(QtCore.QRect(0, 0, 711, 631))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.welcomeLabel = QtWidgets.QLabel(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.welcomeLabel.sizePolicy().hasHeightForWidth())
        self.welcomeLabel.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("汉仪长宋简")
        font.setPointSize(24)
        self.welcomeLabel.setFont(font)
        self.welcomeLabel.setStyleSheet("background-color:rgb(200, 200, 200);")
        self.welcomeLabel.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.welcomeLabel.setFrameShadow(QtWidgets.QFrame.Plain)
        self.welcomeLabel.setTextFormat(QtCore.Qt.AutoText)
        self.welcomeLabel.setObjectName("welcomeLabel")
        self.verticalLayout.addWidget(self.welcomeLabel)
        self.labelline = QtWidgets.QLabel(self.layoutWidget)
        self.labelline.setMaximumSize(QtCore.QSize(16777215, 2))
        self.labelline.setStyleSheet("background-color: rgb(57, 57, 57);")
        self.labelline.setText("")
        self.labelline.setObjectName("labelline")
        self.verticalLayout.addWidget(self.labelline)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame_2 = QtWidgets.QFrame(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(4)
        sizePolicy.setVerticalStretch(5)
        sizePolicy.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy)
        self.frame_2.setStyleSheet("background-color: rgb(238, 238, 238);\n"
"border-top-color: rgb(43, 43, 43);")
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.frame = QtWidgets.QFrame(self.frame_2)
        self.frame.setGeometry(QtCore.QRect(-10, 0, 141, 521))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(5)
        sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy)
        self.frame.setStyleSheet("background-color: rgb(200, 200, 200);")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.pushButton_1 = QtWidgets.QPushButton(self.frame)
        self.pushButton_1.setGeometry(QtCore.QRect(0, 0, 151, 51))
        font = QtGui.QFont()
        font.setFamily("方正隶变_GBK")
        font.setPointSize(12)
        self.pushButton_1.setFont(font)
        self.pushButton_1.setStyleSheet("background-color: rgb(230, 230, 230);")
        self.pushButton_1.setObjectName("pushButton_1")
        self.pushButton_2 = QtWidgets.QPushButton(self.frame)
        self.pushButton_2.setGeometry(QtCore.QRect(0, 50, 151, 51))
        font = QtGui.QFont()
        font.setFamily("方正隶变_GBK")
        font.setPointSize(12)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setStyleSheet("background-color: rgb(210, 210, 210);")
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.frame)
        self.pushButton_3.setGeometry(QtCore.QRect(0, 100, 151, 51))
        font = QtGui.QFont()
        font.setFamily("方正隶变_GBK")
        font.setPointSize(12)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setStyleSheet("background-color: rgb(210, 210, 210);")
        self.pushButton_3.setObjectName("pushButton_3")
        self.label_4 = QtWidgets.QLabel(self.frame)
        self.label_4.setGeometry(QtCore.QRect(10, 150, 141, 41))
        self.label_4.setText("")
        self.label_4.setObjectName("label_4")
        self.frame_3 = QtWidgets.QFrame(self.frame_2)
        self.frame_3.setGeometry(QtCore.QRect(129, -1, 581, 501))
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.stackedWidget = QtWidgets.QStackedWidget(self.frame_3)
        self.stackedWidget.setGeometry(QtCore.QRect(-1, -1, 581, 511))
        self.stackedWidget.setStyleSheet("background-color: rgb(238, 238, 238);")
        self.stackedWidget.setObjectName("stackedWidget")
        self.page_3 = QtWidgets.QWidget()
        self.page_3.setStyleSheet("background-color: rgb(238, 238, 238);")
        self.page_3.setObjectName("page_3")
        self.textEdit = QtWidgets.QTextEdit(self.page_3)
        self.textEdit.setGeometry(QtCore.QRect(40, 70, 491, 91))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(10)
        self.textEdit.setFont(font)
        self.textEdit.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.textEdit.setObjectName("textEdit")
        self.label = QtWidgets.QLabel(self.page_3)
        self.label.setGeometry(QtCore.QRect(40, 30, 131, 31))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.page_3)
        self.label_2.setGeometry(QtCore.QRect(40, 200, 131, 31))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.textEdit_2 = QtWidgets.QTextEdit(self.page_3)
        self.textEdit_2.setGeometry(QtCore.QRect(40, 240, 491, 91))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(10)
        self.textEdit_2.setFont(font)
        self.textEdit_2.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.textEdit_2.setObjectName("textEdit_2")
        self.label_3 = QtWidgets.QLabel(self.page_3)
        self.label_3.setGeometry(QtCore.QRect(40, 370, 131, 31))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.pushButton = QtWidgets.QPushButton(self.page_3)
        self.pushButton.setGeometry(QtCore.QRect(420, 30, 111, 28))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.pushButton.setFont(font)
        self.pushButton.setAutoFillBackground(False)
        self.pushButton.setStyleSheet("#pushButton{\n"
"background-color:rgb(230, 230, 230);\n"
"border:1px solid;\n"
"border-radius:7px;\n"
"}\n"
"\n"
"#pushButton:pressed{\n"
"background-color:rgb(220, 220, 220);\n"
"border:1px solid;\n"
"border-radius:7px}")
        self.pushButton.setObjectName("pushButton")
        self.radioButton = QtWidgets.QRadioButton(self.page_3)
        self.radioButton.setGeometry(QtCore.QRect(130, 410, 101, 19))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.radioButton.setFont(font)
        self.radioButton.setObjectName("radioButton")
        self.radioButton_2 = QtWidgets.QRadioButton(self.page_3)
        self.radioButton_2.setGeometry(QtCore.QRect(350, 410, 101, 19))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.radioButton_2.setFont(font)
        self.radioButton_2.setObjectName("radioButton_2")
        self.stackedWidget.addWidget(self.page_3)
        self.page = QtWidgets.QWidget()
        self.page.setObjectName("page")
        self.listWidget = QtWidgets.QListWidget(self.page)
        self.listWidget.setGeometry(QtCore.QRect(50, 90, 481, 141))
        self.listWidget.setObjectName("listWidget")
        self.label_7 = QtWidgets.QLabel(self.page)
        self.label_7.setGeometry(QtCore.QRect(50, 30, 51, 31))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.lineEdit = mylineedit(self.page)
        self.lineEdit.setGeometry(QtCore.QRect(110, 30, 191, 31))
        self.lineEdit.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.lineEdit.setObjectName("lineEdit")
        self.pushButton_4 = QtWidgets.QPushButton(self.page)
        self.pushButton_4.setGeometry(QtCore.QRect(420, 30, 111, 28))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setAutoFillBackground(False)
        self.pushButton_4.setStyleSheet("#pushButton_4{\n"
"background-color:rgb(230, 230, 230);\n"
"border:1px solid;\n"
"border-radius:7px;\n"
"}\n"
"\n"
"#pushButton_4:pressed{\n"
"background-color:rgb(220, 220, 220);\n"
"border:1px solid;\n"
"border-radius:7px}")
        self.pushButton_4.setObjectName("pushButton_4")
        self.label_8 = QtWidgets.QLabel(self.page)
        self.label_8.setGeometry(QtCore.QRect(310, 30, 21, 31))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.page)
        self.label_9.setGeometry(QtCore.QRect(390, 30, 21, 31))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.spinBox = QtWidgets.QSpinBox(self.page)
        self.spinBox.setGeometry(QtCore.QRect(340, 30, 46, 31))
        self.spinBox.setObjectName("spinBox")
        self.label_10 = QtWidgets.QLabel(self.page)
        self.label_10.setGeometry(QtCore.QRect(330, 340, 91, 31))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.page)
        self.label_11.setGeometry(QtCore.QRect(330, 410, 91, 31))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.page)
        self.label_12.setGeometry(QtCore.QRect(430, 340, 101, 31))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.label_12.setFont(font)
        self.label_12.setText("")
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.page)
        self.label_13.setGeometry(QtCore.QRect(430, 410, 101, 31))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.label_13.setFont(font)
        self.label_13.setText("")
        self.label_13.setObjectName("label_13")
        self.label_17 = QtWidgets.QLabel(self.page)
        self.label_17.setGeometry(QtCore.QRect(330, 270, 91, 31))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.label_17.setFont(font)
        self.label_17.setObjectName("label_17")
        self.label_18 = QtWidgets.QLabel(self.page)
        self.label_18.setGeometry(QtCore.QRect(430, 270, 101, 31))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.label_18.setFont(font)
        self.label_18.setText("")
        self.label_18.setObjectName("label_18")
        self.label_5 = QtWidgets.QLabel(self.page)
        self.label_5.setGeometry(QtCore.QRect(130, 70, 311, 16))
        font = QtGui.QFont()
        font.setFamily("等线")
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.stackedWidget.addWidget(self.page)
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setObjectName("page_2")
        self.listWidget_2 = QtWidgets.QListWidget(self.page_2)
        self.listWidget_2.setGeometry(QtCore.QRect(50, 90, 481, 141))
        self.listWidget_2.setObjectName("listWidget_2")
        self.pushButton_5 = QtWidgets.QPushButton(self.page_2)
        self.pushButton_5.setGeometry(QtCore.QRect(420, 30, 111, 28))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.pushButton_5.setFont(font)
        self.pushButton_5.setAutoFillBackground(False)
        self.pushButton_5.setStyleSheet("#pushButton_5{\n"
"background-color:rgb(230, 230, 230);\n"
"border:1px solid;\n"
"border-radius:7px;\n"
"}\n"
"\n"
"#pushButton_5:pressed{\n"
"background-color:rgb(220, 220, 220);\n"
"border:1px solid;\n"
"border-radius:7px}")
        self.pushButton_5.setObjectName("pushButton_5")
        self.label_16 = QtWidgets.QLabel(self.page_2)
        self.label_16.setGeometry(QtCore.QRect(50, 30, 51, 31))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.label_16.setFont(font)
        self.label_16.setObjectName("label_16")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.page_2)
        self.lineEdit_2.setGeometry(QtCore.QRect(110, 30, 291, 31))
        self.lineEdit_2.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.label_28 = QtWidgets.QLabel(self.page_2)
        self.label_28.setGeometry(QtCore.QRect(430, 340, 101, 31))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.label_28.setFont(font)
        self.label_28.setText("")
        self.label_28.setObjectName("label_28")
        self.label_29 = QtWidgets.QLabel(self.page_2)
        self.label_29.setGeometry(QtCore.QRect(430, 410, 101, 31))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.label_29.setFont(font)
        self.label_29.setText("")
        self.label_29.setObjectName("label_29")
        self.label_30 = QtWidgets.QLabel(self.page_2)
        self.label_30.setGeometry(QtCore.QRect(330, 410, 91, 31))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.label_30.setFont(font)
        self.label_30.setObjectName("label_30")
        self.label_31 = QtWidgets.QLabel(self.page_2)
        self.label_31.setGeometry(QtCore.QRect(330, 270, 91, 31))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.label_31.setFont(font)
        self.label_31.setObjectName("label_31")
        self.label_32 = QtWidgets.QLabel(self.page_2)
        self.label_32.setGeometry(QtCore.QRect(330, 340, 91, 31))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.label_32.setFont(font)
        self.label_32.setObjectName("label_32")
        self.label_33 = QtWidgets.QLabel(self.page_2)
        self.label_33.setGeometry(QtCore.QRect(430, 270, 101, 31))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.label_33.setFont(font)
        self.label_33.setText("")
        self.label_33.setObjectName("label_33")
        self.label_6 = QtWidgets.QLabel(self.page_2)
        self.label_6.setGeometry(QtCore.QRect(130, 70, 311, 16))
        font = QtGui.QFont()
        font.setFamily("等线")
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.stackedWidget.addWidget(self.page_2)
        self.horizontalLayout.addWidget(self.frame_2)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(2, 4)

        self.retranslateUi(Form)
        self.stackedWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "评论分析"))
        self.welcomeLabel.setText(_translate("Form", "欢迎使用亚马逊评论情感分析程序"))
        self.pushButton_1.setText(_translate("Form", "单条评论"))
        self.pushButton_2.setText(_translate("Form", "评论文件"))
        self.pushButton_3.setText(_translate("Form", "商品网址"))
        self.textEdit.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'等线\'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.label.setText(_translate("Form", "输入文本："))
        self.label_2.setText(_translate("Form", "翻译："))
        self.textEdit_2.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'等线\'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.label_3.setText(_translate("Form", "情绪预测："))
        self.pushButton.setText(_translate("Form", "确定"))
        self.radioButton.setText(_translate("Form", "积极"))
        self.radioButton_2.setText(_translate("Form", "消极"))
        self.label_7.setText(_translate("Form", "文件："))
        self.pushButton_4.setText(_translate("Form", "确定"))
        self.label_8.setText(_translate("Form", "第"))
        self.label_9.setText(_translate("Form", "列"))
        self.label_10.setText(_translate("Form", "积极情绪："))
        self.label_11.setText(_translate("Form", "消极情绪："))
        self.label_17.setText(_translate("Form", "评论总数："))
        self.label_5.setText(_translate("Form", "注：双击输入框可选择文件，只可选用csv文件"))
        self.pushButton_5.setText(_translate("Form", "确定"))
        self.label_16.setText(_translate("Form", "网址："))
        self.label_30.setText(_translate("Form", "消极情绪："))
        self.label_31.setText(_translate("Form", "评论总数："))
        self.label_32.setText(_translate("Form", "积极情绪："))
        self.label_6.setText(_translate("Form", "注：请复制评论网址，且只复制到ASIN编号处"))
# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'demo.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1125, 842)
        MainWindow.setMaximumSize(QtCore.QSize(1125, 842))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 0, 1121, 61))
        self.label.setStyleSheet("QFrame\n"
"{\n"
"    color:#ffffff; /*文字颜色*/\n"
"    background-color:qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop:0 #aa55ff, stop: 1 #1296db);/*背景色*/\n"
"    border-style:outset; /*边框风格*/\n"
"    border-width:2px;/*边框宽度*/\n"
"    border-color:#0055ff; /*边框颜色*/\n"
"    border-radius:10px; /*边框倒角*/\n"
"    font:bold 14px; /*字体*/\n"
"    font-family: Segoe UI;\n"
"    min-width:100px;/*控件最小宽度*/\n"
"    min-height:20px;/*控件最小高度*/\n"
"    padding:4px;/*内边距*/\n"
"}")
        self.label.setObjectName("label")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(0, 760, 1121, 32))
        self.frame.setStyleSheet("QFrame\n"
"{\n"
"    color:#ffffff; /*文字颜色*/\n"
"    background-color:qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop:0 #aa55ff, stop: 1 #1296db);/*背景色*/\n"
"    border-style:outset; /*边框风格*/\n"
"    border-width:2px;/*边框宽度*/\n"
"    border-color:#0055ff; /*边框颜色*/\n"
"    border-radius:10px; /*边框倒角*/\n"
"    font:bold 14px; /*字体*/\n"
"    font-family: Segoe UI;\n"
"    min-width:100px;/*控件最小宽度*/\n"
"    min-height:20px;/*控件最小高度*/\n"
"    padding:4px;/*内边距*/\n"
"}")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.ScrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.ScrollArea.setGeometry(QtCore.QRect(310, 60, 811, 701))
        self.ScrollArea.setWidgetResizable(True)
        self.ScrollArea.setObjectName("ScrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 809, 699))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.scrollBar_1 = QtWidgets.QScrollBar(self.scrollAreaWidgetContents)
        self.scrollBar_1.setGeometry(QtCore.QRect(790, 0, 20, 701))
        self.scrollBar_1.setOrientation(QtCore.Qt.Vertical)
        self.scrollBar_1.setObjectName("scrollBar_1")
        self.ScrollArea.setWidget(self.scrollAreaWidgetContents)
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(30, 80, 181, 241))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.Process = QtWidgets.QPushButton(self.layoutWidget)
        self.Process.setStyleSheet("QPushButton\n"
"{\n"
"    color:#ffffff; /*文字颜色*/\n"
"    background-color:qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop:0 #aa55ff, stop: 1 #1296db);/*背景色*/\n"
"    border-style:outset; /*边框风格*/\n"
"    border-width:2px;/*边框宽度*/\n"
"    border-color:#0055ff; /*边框颜色*/\n"
"    border-radius:10px; /*边框倒角*/\n"
"    font:bold 14px; /*字体*/\n"
"    font-family: Segoe UI;\n"
"    min-width:100px;/*控件最小宽度*/\n"
"    min-height:20px;/*控件最小高度*/\n"
"    padding:4px;/*内边距*/\n"
"}\n"
"\n"
"QPushButton::hover{\n"
"    color:#ffffff; /*文字颜色*/\n"
"    background-color:qlineargradient(x1: 1, y1: 1, x2: 1, y2: 0, stop:0 #aa55ff, stop: 1 #1296db);/*背景色*/\n"
"    border-style:outset; /*边框风格*/\n"
"    border-width:2px;/*边框宽度*/\n"
"    border-color:#0055ff; /*边框颜色*/\n"
"    border-radius:10px; /*边框倒角*/\n"
"    font:bold 14px; /*字体*/\n"
"    font-family: Segoe UI;\n"
"    min-width:100px;/*控件最小宽度*/\n"
"    min-height:20px;/*控件最小高度*/\n"
"    padding:4px;/*内边距*/\n"
"}")
        self.Process.setObjectName("Process")
        self.verticalLayout.addWidget(self.Process)
        self.Stop_process = QtWidgets.QPushButton(self.layoutWidget)
        self.Stop_process.setMaximumSize(QtCore.QSize(171, 16777215))
        self.Stop_process.setStyleSheet("QPushButton\n"
"{\n"
"    color:#ffffff; /*文字颜色*/\n"
"    background-color:qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop:0 #aa55ff, stop: 1 #1296db);/*背景色*/\n"
"    border-style:outset; /*边框风格*/\n"
"    border-width:2px;/*边框宽度*/\n"
"    border-color:#0055ff; /*边框颜色*/\n"
"    border-radius:10px; /*边框倒角*/\n"
"    font:bold 14px; /*字体*/\n"
"    font-family: Segoe UI;\n"
"    min-width:100px;/*控件最小宽度*/\n"
"    min-height:20px;/*控件最小高度*/\n"
"    padding:4px;/*内边距*/\n"
"}\n"
"\n"
"QPushButton::hover{\n"
"    color:#ffffff; /*文字颜色*/\n"
"    background-color:qlineargradient(x1: 1, y1: 1, x2: 1, y2: 0, stop:0 #aa55ff, stop: 1 #1296db);/*背景色*/\n"
"    border-style:outset; /*边框风格*/\n"
"    border-width:2px;/*边框宽度*/\n"
"    border-color:#0055ff; /*边框颜色*/\n"
"    border-radius:10px; /*边框倒角*/\n"
"    font:bold 14px; /*字体*/\n"
"    font-family: Segoe UI;\n"
"    min-width:100px;/*控件最小宽度*/\n"
"    min-height:20px;/*控件最小高度*/\n"
"    padding:4px;/*内边距*/\n"
"}")
        self.Stop_process.setObjectName("Stop_process")
        self.verticalLayout.addWidget(self.Stop_process)
        self.Present_images = QtWidgets.QPushButton(self.layoutWidget)
        self.Present_images.setStyleSheet("QPushButton\n"
"{\n"
"    color:#ffffff; /*文字颜色*/\n"
"    background-color:qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop:0 #aa55ff, stop: 1 #1296db);/*背景色*/\n"
"    border-style:outset; /*边框风格*/\n"
"    border-width:2px;/*边框宽度*/\n"
"    border-color:#0055ff; /*边框颜色*/\n"
"    border-radius:10px; /*边框倒角*/\n"
"    font:bold 14px; /*字体*/\n"
"    font-family: Segoe UI;\n"
"    min-width:100px;/*控件最小宽度*/\n"
"    min-height:20px;/*控件最小高度*/\n"
"    padding:4px;/*内边距*/\n"
"}\n"
"\n"
"QPushButton::hover{\n"
"    color:#ffffff; /*文字颜色*/\n"
"    background-color:qlineargradient(x1: 1, y1: 1, x2: 1, y2: 0, stop:0 #aa55ff, stop: 1 #1296db);/*背景色*/\n"
"    border-style:outset; /*边框风格*/\n"
"    border-width:2px;/*边框宽度*/\n"
"    border-color:#0055ff; /*边框颜色*/\n"
"    border-radius:10px; /*边框倒角*/\n"
"    font:bold 14px; /*字体*/\n"
"    font-family: Segoe UI;\n"
"    min-width:100px;/*控件最小宽度*/\n"
"    min-height:20px;/*控件最小高度*/\n"
"    padding:4px;/*内边距*/\n"
"}")
        self.Present_images.setObjectName("Present_images")
        self.verticalLayout.addWidget(self.Present_images)
        self.Integrate = QtWidgets.QPushButton(self.layoutWidget)
        self.Integrate.setStyleSheet("QPushButton\n"
"{\n"
"    color:#ffffff; /*文字颜色*/\n"
"    background-color:qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop:0 #aa55ff, stop: 1 #1296db);/*背景色*/\n"
"    border-style:outset; /*边框风格*/\n"
"    border-width:2px;/*边框宽度*/\n"
"    border-color:#0055ff; /*边框颜色*/\n"
"    border-radius:10px; /*边框倒角*/\n"
"    font:bold 14px; /*字体*/\n"
"    font-family: Segoe UI;\n"
"    min-width:100px;/*控件最小宽度*/\n"
"    min-height:20px;/*控件最小高度*/\n"
"    padding:4px;/*内边距*/\n"
"}\n"
"\n"
"QPushButton::hover{\n"
"    color:#ffffff; /*文字颜色*/\n"
"    background-color:qlineargradient(x1: 1, y1: 1, x2: 1, y2: 0, stop:0 #aa55ff, stop: 1 #1296db);/*背景色*/\n"
"    border-style:outset; /*边框风格*/\n"
"    border-width:2px;/*边框宽度*/\n"
"    border-color:#0055ff; /*边框颜色*/\n"
"    border-radius:10px; /*边框倒角*/\n"
"    font:bold 14px; /*字体*/\n"
"    font-family: Segoe UI;\n"
"    min-width:100px;/*控件最小宽度*/\n"
"    min-height:20px;/*控件最小高度*/\n"
"    padding:4px;/*内边距*/\n"
"}")
        self.Integrate.setObjectName("Integrate")
        self.verticalLayout.addWidget(self.Integrate)
        self.Preview = QtWidgets.QPushButton(self.layoutWidget)
        self.Preview.setStyleSheet("QPushButton\n"
"{\n"
"    color:#ffffff; /*文字颜色*/\n"
"    background-color:qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop:0 #aa55ff, stop: 1 #1296db);/*背景色*/\n"
"    border-style:outset; /*边框风格*/\n"
"    border-width:2px;/*边框宽度*/\n"
"    border-color:#0055ff; /*边框颜色*/\n"
"    border-radius:10px; /*边框倒角*/\n"
"    font:bold 14px; /*字体*/\n"
"    font-family: Segoe UI;\n"
"    min-width:100px;/*控件最小宽度*/\n"
"    min-height:20px;/*控件最小高度*/\n"
"    padding:4px;/*内边距*/\n"
"}\n"
"\n"
"QPushButton::hover{\n"
"    color:#ffffff; /*文字颜色*/\n"
"    background-color:qlineargradient(x1: 1, y1: 1, x2: 1, y2: 0, stop:0 #aa55ff, stop: 1 #1296db);/*背景色*/\n"
"    border-style:outset; /*边框风格*/\n"
"    border-width:2px;/*边框宽度*/\n"
"    border-color:#0055ff; /*边框颜色*/\n"
"    border-radius:10px; /*边框倒角*/\n"
"    font:bold 14px; /*字体*/\n"
"    font-family: Segoe UI;\n"
"    min-width:100px;/*控件最小宽度*/\n"
"    min-height:20px;/*控件最小高度*/\n"
"    padding:4px;/*内边距*/\n"
"}")
        self.Preview.setObjectName("Preview")
        self.verticalLayout.addWidget(self.Preview)
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(30, 520, 251, 201))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_6 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_6.setStyleSheet("self.setStyleSheet(\"QLabel{background:white;}\"\n"
"                   \"QLabel{color:rgb(100,100,100,250);font-size:15px;font-weight:bold;font-family:Roman times;}\"\n"
"                   \"QLabel:hover{color:rgb(100,100,100,120);}\")")
        self.label_6.setObjectName("label_6")
        self.verticalLayout_2.addWidget(self.label_6)
        self.integrate_rate = QtWidgets.QProgressBar(self.verticalLayoutWidget)
        self.integrate_rate.setStyleSheet("QProgressBar::chunk\n"
"{\n"
"border-radius:11px;\n"
"background:qlineargradient(spread:pad,x1:0,y1:0,x2:1,y2:0,stop:0 #01FAFF,stop:1  #26B4FF);\n"
"}\n"
"QProgressBar#progressBar\n"
"{\n"
"height:22px;\n"
"text-align:center;/*文本位置*/\n"
"font-size:14px;\n"
"color:white;\n"
"border-radius:11px;\n"
"background: #1D5573 ;\n"
"}")
        self.integrate_rate.setProperty("value", 0)
        self.integrate_rate.setObjectName("integrate_rate")
        self.verticalLayout_2.addWidget(self.integrate_rate)
        self.label_2 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_2.setStyleSheet("self.setStyleSheet(\"QLabel{background:white;}\"\n"
"                   \"QLabel{color:rgb(100,100,100,250);font-size:15px;font-weight:bold;font-family:Roman times;}\"\n"
"                   \"QLabel:hover{color:rgb(100,100,100,120);}\")")
        self.label_2.setObjectName("label_2")
        self.verticalLayout_2.addWidget(self.label_2)
        self.process_rate = QtWidgets.QProgressBar(self.verticalLayoutWidget)
        self.process_rate.setStyleSheet("QProgressBar::chunk\n"
"{\n"
"border-radius:11px;\n"
"background:qlineargradient(spread:pad,x1:0,y1:0,x2:1,y2:0,stop:0 #01FAFF,stop:1  #26B4FF);\n"
"}\n"
"QProgressBar#progressBar\n"
"{\n"
"height:22px;\n"
"text-align:center;/*文本位置*/\n"
"font-size:14px;\n"
"color:white;\n"
"border-radius:11px;\n"
"background: #1D5573 ;\n"
"}")
        self.process_rate.setProperty("value", 0)
        self.process_rate.setObjectName("process_rate")
        self.verticalLayout_2.addWidget(self.process_rate)
        self.label_3 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_2.addWidget(self.label_3)
        self.clip_rate = QtWidgets.QProgressBar(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.clip_rate.setFont(font)
        self.clip_rate.setStyleSheet("QProgressBar::chunk\n"
"{\n"
"border-radius:11px;\n"
"background:qlineargradient(spread:pad,x1:0,y1:0,x2:1,y2:0,stop:0 #01FAFF,stop:1  #26B4FF);\n"
"}\n"
"QProgressBar#progressBar\n"
"{\n"
"height:22px;\n"
"text-align:center;/*文本位置*/\n"
"font-size:14px;\n"
"color:white;\n"
"border-radius:11px;\n"
"background: #1D5573 ;\n"
"}")
        self.clip_rate.setProperty("value", 0)
        self.clip_rate.setTextVisible(True)
        self.clip_rate.setObjectName("clip_rate")
        self.verticalLayout_2.addWidget(self.clip_rate)
        self.Manual_cut = QtWidgets.QPushButton(self.centralwidget)
        self.Manual_cut.setGeometry(QtCore.QRect(30, 350, 179, 32))
        self.Manual_cut.setStyleSheet("QPushButton\n"
"{\n"
"    color:#ffffff; /*文字颜色*/\n"
"    background-color:qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop:0 #aa55ff, stop: 1 #1296db);/*背景色*/\n"
"    border-style:outset; /*边框风格*/\n"
"    border-width:2px;/*边框宽度*/\n"
"    border-color:#0055ff; /*边框颜色*/\n"
"    border-radius:10px; /*边框倒角*/\n"
"    font:bold 14px; /*字体*/\n"
"    font-family: Segoe UI;\n"
"    min-width:100px;/*控件最小宽度*/\n"
"    min-height:20px;/*控件最小高度*/\n"
"    padding:4px;/*内边距*/\n"
"}\n"
"\n"
"QPushButton::hover{\n"
"    color:#ffffff; /*文字颜色*/\n"
"    background-color:qlineargradient(x1: 1, y1: 1, x2: 1, y2: 0, stop:0 #aa55ff, stop: 1 #1296db);/*背景色*/\n"
"    border-style:outset; /*边框风格*/\n"
"    border-width:2px;/*边框宽度*/\n"
"    border-color:#0055ff; /*边框颜色*/\n"
"    border-radius:10px; /*边框倒角*/\n"
"    font:bold 14px; /*字体*/\n"
"    font-family: Segoe UI;\n"
"    min-width:100px;/*控件最小宽度*/\n"
"    min-height:20px;/*控件最小高度*/\n"
"    padding:4px;/*内边距*/\n"
"}")
        self.Manual_cut.setObjectName("Manual_cut")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(30, 410, 191, 21))
        self.label_4.setObjectName("label_4")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(30, 440, 201, 80))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.start_time = QtWidgets.QLineEdit(self.horizontalLayoutWidget)
        self.start_time.setObjectName("start_time")
        self.horizontalLayout.addWidget(self.start_time)
        self.label_5 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout.addWidget(self.label_5)
        self.end_time = QtWidgets.QLineEdit(self.horizontalLayoutWidget)
        self.end_time.setObjectName("end_time")
        self.horizontalLayout.addWidget(self.end_time)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1125, 26))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen_file = QtWidgets.QAction(MainWindow)
        self.actionOpen_file.setObjectName("actionOpen_file")
        self.menu.addAction(self.actionOpen_file)
        self.menubar.addAction(self.menu.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:16pt;\">Video processor</span></p></body></html>"))
        self.Process.setText(_translate("MainWindow", "Process"))
        self.Stop_process.setText(_translate("MainWindow", "Stop processing"))
        self.Present_images.setText(_translate("MainWindow", "Present the images"))
        self.Integrate.setText(_translate("MainWindow", "Integrate"))
        self.Preview.setText(_translate("MainWindow", "Preview"))
        self.label_6.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-weight:600; color:#aa00ff;\">合并进度</span></p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-weight:600; color:#aa55ff;\">处理进度</span></p></body></html>"))
        self.label_3.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-weight:600; color:#aa55ff;\">手动剪辑进度</span></p></body></html>"))
        self.Manual_cut.setText(_translate("MainWindow", "Manual cut"))
        self.label_4.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-weight:600;\">请填写你需要剪辑的时间范围</span></p></body></html>"))
        self.label_5.setText(_translate("MainWindow", "到"))
        self.menu.setTitle(_translate("MainWindow", "文件"))
        self.actionOpen_file.setText(_translate("MainWindow", "Open file"))
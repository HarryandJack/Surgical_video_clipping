# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'video_1.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1053, 708)
        self.btn_select = QtWidgets.QPushButton(Form)
        self.btn_select.setGeometry(QtCore.QRect(280, 500, 201, 51))
        self.btn_select.setObjectName("btn_select")
        self.wgt_player = QVideoWidget(Form)
        self.wgt_player.setGeometry(QtCore.QRect(100, 60, 911, 351))
        self.wgt_player.setObjectName("wgt_player")
        self.btn_play_pause = QtWidgets.QPushButton(Form)
        self.btn_play_pause.setGeometry(QtCore.QRect(630, 510, 93, 28))
        self.btn_play_pause.setObjectName("btn_play_pause")
        self.sld_duration = QtWidgets.QSlider(Form)
        self.sld_duration.setGeometry(QtCore.QRect(330, 450, 160, 22))
        self.sld_duration.setOrientation(QtCore.Qt.Horizontal)
        self.sld_duration.setObjectName("sld_duration")
        self.lab_duration = QtWidgets.QLabel(Form)
        self.lab_duration.setGeometry(QtCore.QRect(630, 450, 201, 31))
        self.lab_duration.setObjectName("lab_duration")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.btn_select.setText(_translate("Form", "播放视频"))
        self.btn_play_pause.setText(_translate("Form", "Stop"))
        self.lab_duration.setText(_translate("Form", "--/--"))
from PyQt5.QtMultimediaWidgets import QVideoWidget
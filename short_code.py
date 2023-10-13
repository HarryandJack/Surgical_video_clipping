import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox,QProgressBar, QSpacerItem, QSizePolicy, QDialog, QCheckBox, QWidget
from PyQt5.QtWidgets import QFileDialog, QLabel, QGridLayout, QVBoxLayout
from demo import Ui_MainWindow
from scenedetect.video_splitter import split_video_ffmpeg
import os
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl, QTimer
import re
from VideoClipper import VideoClipper
from VideoProcessor import VideoProcessor
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, QThread, Qt
from scenedetect import open_video, ContentDetector, SceneManager
from scenedetect.stats_manager import StatsManager
from moviepy.editor import *
from moviepy.editor import VideoFileClip, concatenate_videoclips
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.video.io.VideoFileClip import VideoFileClip
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtWidgets import QFileDialog, QApplication
from PyQt5 import uic

# Pyqt5预览视频需要添加LAV视频解码器才能播放


# MyWindow 类继承了两个类的功能，一方面它是一个主窗口，拥有主窗口的功能，另一方面它也拥有从 Ui_MainWindow 类继承而来的界面设计
class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.video_player = videoPlayer()  # 创建 videoPlayer 实例
        self.Preview.clicked.connect(self.show_video_player)  # 连接按钮点击事件

    def show_video_player(self):
        self.video_player.show()
class videoPlayer(QWidget):
    def __init__(self):
        super().__init__()
        # 初始化
        self.ui = uic.loadUi('video_1.ui')  # 加载designer设计的ui程序
        # 播放器
        self.player = QMediaPlayer()
        self.player.setVideoOutput(self.ui.wgt_player)
        # 按钮
        self.ui.btn_select.clicked.connect(self.open)
        self.ui.btn_play_pause.clicked.connect(self.playPause)
        # 进度条
        self.player.durationChanged.connect(self.getDuration)
        self.player.positionChanged.connect(self.getPosition)
        self.ui.sld_duration.sliderMoved.connect(self.updatePosition)

    # 打开视频文件
    def open(self):
        file_url = QFileDialog.getOpenFileUrl()
        print(file_url)  # Check if the file path is printed
        if file_url:
            self.player.setMedia(QMediaContent(file_url[0]))
            self.player.play()
    # 播放视频
    def playPause(self):
        if self.player.state()==1:
            self.player.pause()
        else:
            self.player.play()
    # 视频总时长获取
    def getDuration(self, d):
        '''d是获取到的视频总时长（ms）'''
        self.ui.sld_duration.setRange(0, d)
        self.ui.sld_duration.setEnabled(True)
        self.displayTime(d)
    # 视频实时位置获取
    def getPosition(self, p):
        self.ui.sld_duration.setValue(p)
        self.displayTime(self.ui.sld_duration.maximum()-p)
    # 显示剩余时间
    def displayTime(self, ms):
        minutes = int(ms/60000)
        seconds = int((ms-minutes*60000)/1000)
        self.ui.lab_duration.setText('{}:{}'.format(minutes, seconds))
    # 用进度条更新视频位置
    def updatePosition(self, v):
        self.player.setPosition(v)
        self.displayTime(self.ui.sld_duration.maximum()-v)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    MyWindow = MyWindow()
    MyWindow.show()
    sys.exit(app.exec_())
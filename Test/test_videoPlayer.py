import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox,QProgressBar, QSpacerItem, QSizePolicy, QDialog, QCheckBox, QWidget
from PyQt5.QtWidgets import QFileDialog, QLabel, QGridLayout, QVBoxLayout
from demo import Ui_MainWindow
from video_1 import Ui_Form
from mainWindow import *
from childWindow import *
from PyQt5.QtGui import QPixmap, QPainter
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


# mainWindow
class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyMainWindow, self).__init__()
        self.setupUi(self)

        self.setGeometry(0, 0, 1024, 600)
        self.setWindowTitle('main window')

    def paintEvent(self, event):
        painter = QPainter(self)
        pixmap = QPixmap("./image/bg.jpg")
        painter.drawPixmap(self.rect(), pixmap)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()


class videoPlayer(QWidget, Ui_Form):
    def __init__(self):
        super().__init__()
        uic.loadUi('video_1.ui', self)  # Load the UI file and set it up
        # 播放器
        self.player = QMediaPlayer()
        self.player.setVideoOutput(self.wgt_player)  # Removed self.ui
        # 按钮
        self.btn_select.clicked.connect(self.open)  # Removed self.ui
        self.btn_play_pause.clicked.connect(self.playPause)  # Removed self.ui
        # 进度条
        self.player.durationChanged.connect(self.getDuration)
        self.player.positionChanged.connect(self.getPosition)
        self.sld_duration.sliderMoved.connect(self.updatePosition)  # Removed self.ui

    # 打开视频文件
    def open(self):
        self.player.setMedia(QMediaContent(QFileDialog.getOpenFileUrl()[0]))
        self.player.play()

    def playPause(self):
        if self.player.state() == 1:
            self.player.pause()
        else:
            self.player.play()

    def getDuration(self, d):
        self.sld_duration.setRange(0, d)  # Note: removed self.ui

    def getPosition(self, p):
        self.sld_duration.setValue(p)  # Note: removed self.ui
        self.displayTime(self.sld_duration.maximum() - p)

    def displayTime(self, ms):
        minutes = int(ms / 60000)
        seconds = int((ms - minutes * 60000) / 1000)
        self.lab_duration.setText('{}:{}'.format(minutes, seconds))  # Note: removed self.ui

    def updatePosition(self, v):
        self.player.setPosition(v)
        self.displayTime(self.sld_duration.maximum() - v)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = MyMainWindow()

    video = videoPlayer()

    btn = main.pushButton  # 主窗体按钮事件绑定
    btn.clicked.connect(video.show)

    main.show()
    sys.exit(app.exec_())


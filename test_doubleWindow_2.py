from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QMainWindow, QWidget
from moviepy.editor import *
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtWidgets import QFileDialog, QApplication
from PyQt5 import uic


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("主窗口")
        button = QPushButton("弹出子窗", self)
        button.clicked.connect(self.show_child)
        self.videoPlayer = videoPlayer()

    def show_child(self):
        self.videoPlayer.show()


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
        self.player.setMedia(QMediaContent(QFileDialog.getOpenFileUrl()[0]))
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


# 运行主窗口
if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = Main()
    window.show()

    sys.exit(app.exec_())
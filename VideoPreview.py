from PyQt5.QtWidgets import QWidget, QApplication, QFileDialog
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5 import uic

class VideoPlayer(QWidget):
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


if __name__ == "__main__":
    app = QApplication([])
    myPlayer = VideoPlayer()
    myPlayer.show()
    app.exec()

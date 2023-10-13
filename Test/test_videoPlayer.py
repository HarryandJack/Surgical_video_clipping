import sys
import cv2
from PyQt5.QtWidgets import QApplication, QDialog, QLabel, QVBoxLayout, QPushButton
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl

class VideoPreviewDialog(QDialog):
    def __init__(self, video_path):
        super().__init__()

        self.setWindowTitle("视频预览")
        self.setGeometry(100, 100, 640, 480)

        self.video_path = video_path

        self.video_widget = QVideoWidget(self)
        self.video_widget.setGeometry(10, 10, 620, 400)

        self.play_button = QPushButton("播放", self)
        self.play_button.setGeometry(10, 420, 100, 30)
        self.play_button.clicked.connect(self.play_video)

        self.pause_button = QPushButton("暂停", self)
        self.pause_button.setGeometry(120, 420, 100, 30)
        self.pause_button.clicked.connect(self.pause_video)

        self.stop_button = QPushButton("停止", self)
        self.stop_button.setGeometry(230, 420, 100, 30)
        self.stop_button.clicked.connect(self.stop_video)

        self.close_button = QPushButton("关闭", self)
        self.close_button.setGeometry(340, 420, 100, 30)
        self.close_button.clicked.connect(self.close)

        self.media_player = QMediaPlayer()
        self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(self.video_path)))
        self.media_player.setVideoOutput(self.video_widget)

    def play_video(self):
        self.media_player.play()

    def pause_video(self):
        self.media_player.pause()

    def stop_video(self):
        self.media_player.stop()

    def closeEvent(self, event):
        self.media_player.stop()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    video_path = "D:\\Postgra\\Throat surgery clip.mp4"  # Replace with your video path
    preview_dialog = VideoPreviewDialog(video_path)
    preview_dialog.show()
    sys.exit(app.exec_())

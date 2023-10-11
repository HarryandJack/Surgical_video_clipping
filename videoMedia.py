from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton, QSlider, QWidget, QVBoxLayout, QLabel, QStyle
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent, QVideoFrame, QVideoSurfaceFormat
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import Qt, QUrl

class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.media_player = QMediaPlayer(self)
        self.video_widget = QVideoWidget(self)
        self.media_player.setVideoOutput(self.video_widget)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)  # Initialize with range 0
        self.slider.sliderMoved.connect(self.set_position)

        layout = QVBoxLayout()
        layout.addWidget(self.video_widget)
        layout.addWidget(self.slider)

        self.play_button = QPushButton('Play')
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_button.clicked.connect(self.play)

        layout.addWidget(self.play_button)

        self.media_player.durationChanged.connect(self.duration_changed)
        self.media_player.positionChanged.connect(self.position_changed)

        widget = QWidget(self)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def open_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        file_path, _ = QFileDialog.getOpenFileName(None, "Open Video File",
                                                  "", "Video Files (*.mp4 *.avi);;All Files (*)", options=options)

        if file_path:
            self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(file_path)))

    def play(self):
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        else:
            self.media_player.play()
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

    def set_position(self, position):
        self.media_player.setPosition(position)

    def duration_changed(self, duration):
        self.slider.setRange(0, duration)

    def position_changed(self, position):
        self.slider.setValue(position)

if __name__ == '__main__':
    app = QApplication([])
    window = VideoPlayer()
    window.show()
    app.exec_()

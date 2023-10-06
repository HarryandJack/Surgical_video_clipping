from PyQt5.QtCore import QThread, pyqtSignal
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


# 定义了一个名为 VideoClipper 的类，它继承自 QThread，这是 PyQt 框架中用于创建线程的基类
class VideoClipper(QThread):
    finished = pyqtSignal()
    progressChanged = pyqtSignal(int)

    def __init__(self, file_path, output_file, start_time, end_time):
        super().__init__()
        self.input_file = file_path
        self.output_file = output_file
        self.start_time = start_time
        self.end_time = end_time

    def run(self):
        video_clip = VideoFileClip(self.input_file)
        start_time_seconds = self.start_time
        end_time_seconds = self.end_time
        output_file_path = self.output_file

        ffmpeg_extract_subclip(self.input_file, start_time_seconds, end_time_seconds, targetname=output_file_path)

        self.finished.emit()
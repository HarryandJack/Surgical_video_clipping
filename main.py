import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox,QProgressBar ,QLabel, QSpacerItem, QSizePolicy, QDialog, QVBoxLayout, QWidget
from demo import Ui_MainWindow
import os
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QThread
from PyQt5.uic import loadUi
import subprocess
import scenedetect
from scenedetect import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.video.io.VideoFileClip import VideoFileClip

# 定义了一个名为 VideoClipper 的类，它继承自 QThread，这是 PyQt 框架中用于创建线程的基类/
class VideoClipper(QThread):

    # 用于创建一个信号（signal），当视频剪辑完成时会发射这个信号
    finished = pyqtSignal()

    # 同样是一个类变量，用于创建一个信号，当视频剪辑的进度发生变化时会发射这个信号，传递一个整数参数表示进度
    progressChanged = pyqtSignal(int)

    # 这是 VideoClipper 类的构造函数。它接受一个参数 file_path，表示要剪辑的视频文件的路径
    def __init__(self, file_path, output_file, start_time, end_time):
        super().__init__()
        self.input_file = file_path
        self.output_file = output_file
        self.start_time = start_time
        self.end_time = end_time

    def run(self):
        # 通过 moviepy 库的 VideoFileClip 类加载了输入的视频文件 (self.input_file)。这一行将视频文件加载到一个变量 video_clip 中
        video_clip = VideoFileClip(self.input_file)

        # Calculate start and end times in seconds
        start_time_seconds = self.start_time
        end_time_seconds = self.end_time

        # Define the output file path
        output_file_path = self.output_file

        # 用于从输入的视频文件中截取一个子片段。具体来说，它接受输入文件路径、开始时间（以秒为单位）、结束时间（以秒为单位）和目标文件路径作为参数。截取后的视频将保存在目标文件路径 (output_file_path)
        ffmpeg_extract_subclip(self.input_file, start_time_seconds, end_time_seconds, targetname=output_file_path)

        # 一旦视频截取完成，发射了一个自定义的信号 finished，这个信号可以被其他部分的代码捕获并进行相应的处理。
        self.finished.emit()

class ImagePresent(QThread):
    finished = pyqtSignal()
    progressChanged = pyqtSignal(int)

# 一个名为 VideoProcessor 的类，继承自 QThread，用于处理视频
class VideoProcessor(QThread):
    finished = pyqtSignal()
    progressChanged = pyqtSignal(int)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        # Create a video manager and a scene manager.
        video_manager = scenedetect.VideoManager([self.file_path])
        scene_manager = scenedetect.SceneManager()

        # Add a detector (ContentDetector) to the scene manager.
        scene_manager.add_detector(scenedetect.detectors.ContentDetector())

        try:
            video_manager.set_downscale_factor()
            video_manager.start()
            scene_manager.detect_scenes(frame_source=video_manager)

            for i, scene in enumerate(scene_manager.get_scene_list()):
                start_frame, end_frame = scene

                if end_frame - start_frame >= 100:  # Only process scenes longer than 100 frames.
                    frame_count = (start_frame + end_frame) // 2
                    self.progressChanged.emit(frame_count)

                    # Process the frames here (e.g., save images).
                    # ...

        finally:
            video_manager.release()

        self.finished.emit()

class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.actionOpen_file.triggered.connect(self.open_file_dialog)
        self.Process.clicked.connect(self.process_video)
        self.Stop_process.clicked.connect(self.stop_processing)
        self.file_path = None
        self.video_processor = None
        self.Auto_cut.clicked.connect(self.clip_video)


    def open_file_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        self.file_path, _ = QFileDialog.getOpenFileName(None, "Open Video File",
        "", "Video Files (*.mp4 *.avi);;All Files (*)", options=options)

        if self.file_path:
            video = cv2.VideoCapture(self.file_path)

            QMessageBox.information(None, "成功导入", f"已成功导入视频文件：{self.file_path}", QMessageBox.Ok)
        else:
            QMessageBox.warning(None, "导入失败", "未选择文件或文件无效", QMessageBox.Ok)


    def process_video(self):
        if self.file_path:
            self.video_processor = VideoProcessor(self.file_path)
            self.video_processor.finished.connect(self.process_finished)
            self.video_processor.start()

        else:
            QMessageBox.warning(None, "处理失败", "未选择文件或文件无效", QMessageBox.Ok)

    def process_finished(self):
        QMessageBox.information(None, "处理完成", "视频处理完成", QMessageBox.Ok)

    def stop_processing(self):
        if self.video_processor and self.video_processor.isRunning():
            self.video_processor.terminate()

    def clip_video(self):
        input_file = self.file_path
        if input_file:
            output_file, _ = QFileDialog.getSaveFileName(None, "Save Video File", "",
                                                         "Video Files (*.mp4);;All Files (*)")
            if output_file:
                start_time = 60
                end_time = 120

                # 创建一个名为 `self.video_clipper` 的 `VideoClipper` 对象，
                # 用于执行视频剪辑操作。传递了输入文件路径、输出文件路径以及剪辑的起始和结束时间。
                self.video_clipper = VideoClipper(input_file, output_file, start_time, end_time)

                # 建立了一个信号-槽连接。progressChanged 信号是 VideoClipper 类中定义的用于传递剪辑进度的信号，
                # 它连接到了 self.update_progress 方法，以便在剪辑过程中更新进度条。
                self.video_clipper.progressChanged.connect(self.update_progress)
                # 建立了一个信号-槽连接。`finished` 信号在剪辑完成时发射，
                # 连接到了 `self.clip_finished` 方法，以便在剪辑完成时执行相应的操作。
                self.video_clipper.finished.connect(self.clip_finished)

                # 开始执行视频剪辑，启动了一个新的线程来处理剪辑操作。
                self.video_clipper.start()

    def update_progress(self, progress):
        self.clip_rate.setValue(progress)

    def clip_finished(self):
        print("剪辑完成")  # 添加这行
        self.clip_rate.setValue(100)
        QMessageBox.information(self, "剪辑完成", "视频剪辑完成", QMessageBox.Ok)
        self.Process.setEnabled(True)  # 启用处理按钮
        self.Stop_process.setEnabled(True)  # 启用停止按钮

    def update_process_progress(self, progress):
        self.process_rate.setValue(progress)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MyWindow = MyWindow()
    MyWindow.show()
    sys.exit(app.exec_())

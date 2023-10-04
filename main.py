import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox,QProgressBar ,QLabel, QSpacerItem, QSizePolicy, QDialog, QVBoxLayout, QWidget
from demo import Ui_MainWindow
from scenedetect.video_splitter import split_video_ffmpeg
import os
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QThread
from PyQt5.uic import loadUi
import subprocess
import scenedetect
from scenedetect import open_video, ContentDetector, SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect import VideoManager
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

    def __init__(self, file_path, csv_path):  # 添加 csv_path 参数
        super().__init__()
        self.file_path = file_path
        self.csv_path = csv_path  # 存储传递进来的 csv_path


    def run(self):
        # scenes 是一个列表，其中包含了从视频中检测到的场景信息,每个场景被表示为一个元组，包含了两个时间码对象，分别是 start_timecode 和 end_timecode
        video = open_video(self.file_path)

        scene_manager = SceneManager(stats_manager=StatsManager())

        content_detector = ContentDetector()

        scene_manager.add_detector(content_detector)

        scene_manager.detect_scenes(video=video)

        scene_list = scene_manager.get_scene_list()

        scene_ranges = [(start_frame, end_frame) for start_frame, end_frame in scene_list]

        output_directory = os.path.join(os.getcwd(), 'images')  # Output directory: current working directory/images

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        output_file_template = os.path.join(output_directory, '$VIDEO_NAME-Scene-$SCENE_NUMBER.mp4')

        split_video_ffmpeg(self.file_path, scene_ranges, output_file_template=output_file_template)

        STATS_FILE_PATH = self.csv_path  # 使用传递进来的 csv_path

        # Save per-frame statistics to disk.
        scene_manager.stats_manager.save_to_csv(csv_file=STATS_FILE_PATH)

        self.finished.emit()

# MyWindow 类继承了两个类的功能，一方面它是一个主窗口，拥有主窗口的功能，另一方面它也拥有从 Ui_MainWindow 类继承而来的界面设计
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
        self.Process.setEnabled(False)  # 禁用处理按钮
        if self.file_path:
            csv_path, _ = QFileDialog.getSaveFileName(None, "Save CSV File", "",
                                                      "CSV Files (*.csv);;All Files (*)")
            if csv_path:  # 检查用户是否选择了 CSV 文件
                self.video_processor = VideoProcessor(self.file_path, csv_path)  # 传递 csv_path
                self.video_processor.progressChanged.connect(self.update_progress)
                self.video_processor.finished.connect(self.process_finished)
                self.video_processor.start()
            else:
                QMessageBox.warning(None, "处理失败", "未选择保存的 CSV 文件", QMessageBox.Ok)
        else:
            QMessageBox.warning(None, "处理失败", "未选择文件或文件无效", QMessageBox.Ok)

    def process_finished(self):
        self.process_rate.setValue(100)
        QMessageBox.information(None, "处理完成", "视频处理完成", QMessageBox.Ok)
        self.Process.setEnabled(True)  # 启用处理按钮

    def stop_processing(self):
        if self.video_processor and self.video_processor.isRunning():
            self.video_processor.terminate()
            self.Process.setEnabled(True)  # 启用处理按钮

    def clip_video(self):
        self.Auto_cut.setEnabled(False)
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

    # 在Python中，所有方法的第一个参数通常都是self，它表示类的实例本身
    # 在类的方法中，self代表当前的类实例。通过使用self，可以访问类中的属性和其他方法。
    def update_process_progress(self, progress):
        self.process_rate.setValue(progress)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MyWindow = MyWindow()
    MyWindow.show()
    sys.exit(app.exec_())

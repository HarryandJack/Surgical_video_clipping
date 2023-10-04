import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox,QProgressBar ,QLabel, QSpacerItem, QSizePolicy, QDialog, QVBoxLayout, QWidget
from PyQt5.QtWidgets import QFileDialog, QLabel
from PyQt5.QtGui import QPixmap
from demo import Ui_MainWindow
from scenedetect.video_splitter import split_video_ffmpeg
import os
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QThread, Qt
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

class ImagePresenter(QThread):
    def run(self):
        try:
            # 创建了一个 QFileDialog 的选项对象。
            options = QFileDialog.Options()

            # 将选项设置为只显示目录
            options |= QFileDialog.ShowDirsOnly

            # 打开一个文件对话框，让用户选择一个目录。getExistingDirectory 返回用户选择的目录路径，并将其存储在 folder_path 变量中
            folder_path = QFileDialog.getExistingDirectory(None, "Select Folder", options=options)

            if folder_path:
                # 遍历选定目录下的所有文件和子目录。
                for filename in os.listdir(folder_path):

                    if filename.endswith('.jpg') or filename.endswith('.png'):
                        # 构建完整的图片文件路径
                        image_path = os.path.join(folder_path, filename)
                        # 创建一个 QLabel 对象，用于显示图片
                        label = QLabel()
                        # 使用 QPixmap 类加载图片，创建一个图片对象
                        pixmap = QPixmap(image_path)

                        # 将 QPixmap 对象设置为 QLabel 的显示内容
                        label.setPixmap(pixmap)

                        # 设置 QLabel 中的图片在水平和垂直方向上都居中显示
                        label.setAlignment(Qt.AlignCenter)

                        # 将 QLabel 添加到一个滚动区域的布局中
                        self.scrollAreaWidgetContents.layout().addWidget(label)

        except Exception as e:
            print(f"Error in ImagePresenter: {e}")


# 一个名为 VideoProcessor 的类，继承自 QThread，用于处理视频
class VideoProcessor(QThread):
    finished = pyqtSignal()
    progressChanged = pyqtSignal(int)

    def __init__(self, file_path, csv_path):
        super().__init__()
        self.file_path = file_path
        self.csv_path = csv_path

    def capture_representative_frame(self, video_path):
        if not os.path.exists(os.path.join(video_path, 'images')):
            os.makedirs(os.path.join(video_path, 'images'))

        for filename in os.listdir(video_path):
            if filename.endswith('.mp4') or filename.endswith('.avi'):
                video_file_path = os.path.join(video_path, filename)

                cap = cv2.VideoCapture(video_file_path)
                if not cap.isOpened():
                    print(f"无法打开视频文件 {filename}")
                    continue

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                ret, frame = cap.read()
                if not ret:
                    print(f"无法读取视频帧 {filename}")
                    continue

                image_path = os.path.join(video_path, 'images', f'{filename}_frame.jpg')
                cv2.imwrite(image_path, frame)

                cap.release()

    def run(self):
        video = open_video(self.file_path)

        scene_manager = SceneManager(stats_manager=StatsManager())

        content_detector = ContentDetector()

        scene_manager.add_detector(content_detector)

        scene_manager.detect_scenes(video=video)

        scene_list = scene_manager.get_scene_list()

        scene_ranges = [(start_frame, end_frame) for start_frame, end_frame in scene_list]

        output_directory_base = 'video'
        output_directory = output_directory_base
        suffix = 1

        while os.path.exists(output_directory):
            suffix += 1
            output_directory = f'{output_directory_base}_{suffix}'

        os.makedirs(output_directory)

        output_file_template = os.path.join(output_directory, '$VIDEO_NAME-Scene-$SCENE_NUMBER.mp4')

        split_video_ffmpeg(self.file_path, scene_ranges, output_file_template=output_file_template)

        STATS_FILE_PATH = self.csv_path
        scene_manager.stats_manager.save_to_csv(csv_file=STATS_FILE_PATH)

        video_path = output_directory
        self.capture_representative_frame(video_path)

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
        self.Manual_cut.clicked.connect(self.clip_video)
        self.Present_images.clicked.connect(self.present_images)
        self.ScrollArea = self.scrollAreaWidgetContents


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

    def present_images(self):  # Renamed the method
        image_presentation = ImagePresenter()  # Create an instance of ImagePresenter
        image_presentation.finished.connect(self.image_presentation_finished)
        image_presentation.start()

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
        self.Manual_cut.setEnabled(False)
        input_file = self.file_path
        if input_file:
            output_file, _ = QFileDialog.getSaveFileName(None, "Save Video File", "",
                                                         "Video Files (*.mp4);;All Files (*)")
            if output_file:
                start_time = self.start_time.text() # Get the text from the QLineEdit
                end_time = self.end_time.text()  # Get the text from the QLineEdit

                try:
                    # Validate user input (make sure they are in the format hh:mm:ss)
                    start_time = [int(x) for x in start_time.split(':')]
                    end_time = [int(x) for x in end_time.split(':')]
                    if not (0 <= start_time[0] < 24 and 0 <= start_time[1] < 60 and 0 <= start_time[2] < 60) or \
                            not (0 <= end_time[0] < 24 and 0 <= end_time[1] < 60 and 0 <= end_time[2] < 60):
                        raise ValueError("Invalid time format")

                    start_seconds = start_time[0] * 3600 + start_time[1] * 60 + start_time[2]
                    end_seconds = end_time[0] * 3600 + end_time[1] * 60 + end_time[2]

                    self.video_clipper = VideoClipper(input_file, output_file, start_seconds, end_seconds)
                    self.video_clipper.progressChanged.connect(self.update_progress)
                    self.video_clipper.finished.connect(self.clip_finished)
                    self.video_clipper.start()

                except ValueError as e:
                    QMessageBox.warning(self, "Invalid Input", "Please enter a valid time span in the format hh:mm:ss.",
                                        QMessageBox.Ok)

        self.Manual_cut.setEnabled(True)

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

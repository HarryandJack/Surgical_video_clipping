import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox,QProgressBar, QSpacerItem, QSizePolicy, QDialog, QCheckBox, QWidget
from PyQt5.QtWidgets import QFileDialog, QLabel, QGridLayout, QVBoxLayout
from demo import Ui_MainWindow
from video_1 import Ui_Form
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
        self.actionOpen_file.triggered.connect(self.open_file_dialog)
        self.Process.clicked.connect(self.process_video)
        self.Stop_process.clicked.connect(self.stop_processing)
        self.file_path = None
        self.video_processor = None
        self.Manual_cut.clicked.connect(self.clip_video)
        self.Present_images.clicked.connect(self.present_images)
        finished = pyqtSignal()
        # 添加两字典，用于跟踪用户选择的图像和视频路径和相应复选框状态
        self.selected_images = {}
        self.selected_videos = {}
        """
        By ensuring that the self.Integrate.clicked.connect(self.integrate_videos) line comes after the creation of self.selected_videos,
        you make sure that the dictionary is available when self.integrate_videos is called.
        """
        self.integrated_video_path = None
        self.Integrate.clicked.connect(self.integrate_videos)
        self.video_player = videoPlayer()  # 创建 videoPlayer 实例
        self.Preview.clicked.connect(self.show_video_player)  # 连接按钮点击事件

    def show_video_player(self):
        self.video_player.show()

    def count_images_in_folder(self, folder_path):
        # 获取指定文件夹内的所有文件和子目录
        files = os.listdir(folder_path)

        # 使用列表推导式筛选出所有以 '.jpg' 或 '.png' 结尾的文件
        images = [f for f in files if f.endswith('.jpg') or f.endswith('.png')]

        # 返回图片数量
        return len(images)



    def get_corresponding_video_path(self, image_path):
        """
        获取与给定图像路径对应的视频路径。

        参数:
            image_path (str): 图像的文件路径。

        返回:
            str: 与给定图像对应的视频文件路径。
        """
        # 获取图片文件名，例如 test-3-Scene-001.mp4_frame.jpg
        image_filename = os.path.basename(image_path)

        # 从图片文件名中提取视频文件名，例如 test-3-Scene-001.mp4
        video_filename = image_filename.split('.mp4_frame.jpg')[0] + '.mp4'

        # 获取上一级目录的路径
        parent_directory = os.path.dirname(os.path.dirname(image_path))

        # 拼接成视频的完整路径
        video_path = os.path.join(parent_directory, video_filename)

        return video_path

    def image_checkbox_changed(self, image_path, state):
        """
        当与图像相关联的复选框状态发生变化时，调用此函数。

        参数:
            image_path (str): 图像的文件路径。
            state (int): 复选框的状态。可以是 Qt.Checked 或 Qt.Unchecked。
        """
        if state == Qt.Checked:
            # 如果复选框被选中，更新 selected_images 字典以将此图像标记为选中状态。
            self.selected_images[image_path] = True

            # 找到与此图像对应的视频路径。
            video_path = self.get_corresponding_video_path(image_path)

            if video_path:
                # 如果找到相应的视频路径，则在 selected_videos 字典中将其标记为选中状态。
                self.selected_videos[video_path] = True
        else:
            # 如果复选框未选中，更新 selected_images 字典以将此图像标记为未选中状态。
            self.selected_images[image_path] = False

            # 找到与此图像对应的视频路径。
            video_path = self.get_corresponding_video_path(image_path)

            if video_path:
                # 如果找到相应的视频路径，则在 selected_videos 字典中将其标记为未选中状态。
                self.selected_videos[video_path] = False

    def present_images(self):
        try:
            # 创建一个 QFileDialog 的选项对象。
            options = QFileDialog.Options()

            # 将选项设置为只显示目录
            options |= QFileDialog.ShowDirsOnly

            # 打开一个文件对话框，让用户选择一个目录。getExistingDirectory 返回用户选择的目录路径，并将其存储在 folder_path 变量中
            folder_path = QFileDialog.getExistingDirectory(None, "Select Folder", options=options)

            # 获取文件夹中的图片数量
            image_count = self.count_images_in_folder(folder_path)
            print(f'The folder contains {image_count} images')

            if folder_path:
                # 创建一个 widget 作为容器
                widget = QWidget()

                # 创建一个网格布局
                layout = QGridLayout(widget)

                # 遍历文件夹内的文件
                for i, filename in enumerate(os.listdir(folder_path)):
                    if filename.endswith('.jpg') or filename.endswith('.png'):

                        # 构建完整的图片文件路径
                        image_path = os.path.join(folder_path, filename)

                        # 创建一个 QLabel 对象
                        label = QLabel()

                        # 使用 QPixmap 类加载图片
                        pixmap = QPixmap(image_path)

                        # 设置图片宽度为150像素，高度等比例缩放
                        pixmap = pixmap.scaledToWidth(150)

                        # 将 QPixmap 对象设置为 QLabel 的显示内容
                        label.setPixmap(pixmap)

                        # 设置 QLabel 中的图片在水平和垂直方向上都居中显示
                        label.setAlignment(Qt.AlignCenter)

                        # 将 label 添加到布局中,i // 5 表示行数，i % 5 表示列数
                        layout.addWidget(label, i // 5, i % 5)

                        checkbox = QCheckBox()  # 创建一个复选框对象

                        # 将复选框的状态变化信号连接到指定的槽函数
                        # 使用 lambda 表达式将 image_path 参数传递给槽函数，确保在状态变化时可以获取到对应的图片路径
                        # Lambda 表达式（也称为匿名函数）是一种在 Python 中创建小型、简单函数的方式，lambda arguments: expression
                        checkbox.stateChanged.connect(
                            lambda state, path=image_path: self.image_checkbox_changed(path, state))

                        # 将复选框添加到布局中，i // 5 表示行数，i % 5 表示列数
                        vbox = QVBoxLayout()

                        # 将标签和复选框添加到垂直布局中
                        vbox.addWidget(label)
                        vbox.addWidget(checkbox)

                        # 将垂直布局添加到网格布局中
                        layout.addLayout(vbox, i // 5, i % 5)

                # 将 widget 设置为滚动区域的子部件
                self.ScrollArea.setWidget(widget)

        except Exception as e:
            print(f"Error in ImagePresenter: {e}")

    def integrate_videos(self):
        self.Integrate.setEnabled(False)

        selected_videos_paths = [path for path, selected in self.selected_videos.items() if selected]

        # 这段代码是测试哪里错了的
        for video_path in selected_videos_paths:
            if not os.path.exists(video_path):
                print(f"Error: 文件 {video_path} 不存在")
                return  # 如果发现文件不存在，直接返回，不继续执行后续代码

        if selected_videos_paths:
            output_file, _ = QFileDialog.getSaveFileName(None, "Save Video File", "",
                                                         "Video Files (*.mp4);;All Files (*)")
            if output_file:
                # 生成 VideoFileClip 对象列表
                video_clips = [VideoFileClip(video_path, audio=True) for video_path in selected_videos_paths]

                # 将视频剪辑整合在一起
                final_clip = concatenate_videoclips(video_clips, method="compose")

                # 将整合后的视频保存到指定路径
                final_clip.write_videofile(output_file, fps=24)  # 可以调整 fps

                self.integrated_video_path = output_file

                # 提示用户整合完成
                reply = QMessageBox.question(self, "整合完成", "视频整合完成，是否要打开合并好的视频？",
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.Yes:
                    # 打开合并好的视频
                    os.system(f'start {output_file}')

        else:
            QMessageBox.warning(None, "整合失败", "未选择视频", QMessageBox.Ok)
            self.Integrate.setEnabled(True)



    def integration_finished(self):
        print("整合完成")
        self.clip_rate.setValue(100)
        QMessageBox.information(self, "整合完成", "视频整合完成", QMessageBox.Ok)
        self.Integrate.setEnabled(True)

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
                self.video_processor.progressChanged.connect(self.update_process_progress)
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
                    start_time = [int(x) for s in re.split(':|：|,|\.', start_time) for x in s.split()]
                    end_time = [int(x) for s in re.split(':|：|,|\.', end_time) for x in s.split()]
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
    MyWindow = MyWindow()
    VideoWindow = videoPlayer()
    MyWindow.show()
    sys.exit(app.exec_())

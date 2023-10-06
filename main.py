import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QCheckBox, QWidget, QFileDialog, QLabel, QGridLayout
from demo import Ui_MainWindow
from scenedetect.video_splitter import split_video_ffmpeg
import os
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, QThread, Qt
from scenedetect import open_video, ContentDetector, SceneManager
from scenedetect.stats_manager import StatsManager
from moviepy.editor import *
from moviepy.editor import VideoFileClip, concatenate_videoclips
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


# 一个名为 VideoProcessor 的类，继承自 QThread，用于处理视频
class VideoProcessor(QThread):
    finished = pyqtSignal()
    progressChanged = pyqtSignal(int)

    def __init__(self, file_path, csv_path):
        super().__init__()
        self.file_path = file_path
        self.csv_path = csv_path

    def capture_representative_frame(self, video_path):
        # 检查是否存在 'images' 文件夹，如果不存在则创建
        if not os.path.exists(os.path.join(video_path, 'images')):
            os.makedirs(os.path.join(video_path, 'images'))

        video_path_list = []

        # 遍历目录下的视频文件
        for filename in os.listdir(video_path):
            if filename.endswith('.mp4') or filename.endswith('.avi'):
                video_file_path = os.path.join(video_path, filename)

                video_path_list.append(video_file_path)
                # 打开视频文件
                cap = cv2.VideoCapture(video_file_path)
                if not cap.isOpened():
                    print(f"无法打开视频文件 {filename}")
                    continue

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # 读取第一帧
                ret, frame = cap.read()
                if not ret:
                    print(f"无法读取视频帧 {filename}")
                    continue

                # 生成图像路径并保存
                image_path = os.path.join(video_path, 'images', f'{filename}_frame.jpg')
                cv2.imwrite(image_path, frame)

                # 释放视频对象
                cap.release()

    def run(self):
        # 打开视频文件
        video = open_video(self.file_path)

        # 创建一个场景管理器，并传入统计管理器作为参数
        scene_manager = SceneManager(stats_manager=StatsManager())

        # 创建一个内容检测器
        content_detector = ContentDetector()

        # 将内容检测器添加到场景管理器中
        scene_manager.add_detector(content_detector)

        # 在视频中检测场景
        scene_manager.detect_scenes(video=video)

        # 获取场景列表
        scene_list = scene_manager.get_scene_list()

        # 将场景起止帧数整理成列表
        scene_ranges = [(start_frame, end_frame) for start_frame, end_frame in scene_list]

        # 设置输出目录和模板
        output_directory_base = 'video'
        output_directory = output_directory_base
        suffix = 1

        # 确保输出目录是唯一的
        while os.path.exists(output_directory):
            suffix += 1
            output_directory = f'{output_directory_base}_{suffix}'

        # 创建输出目录
        os.makedirs(output_directory)

        # 设置输出文件模板
        output_file_template = os.path.join(output_directory, '$VIDEO_NAME-Scene-$SCENE_NUMBER.mp4')

        # 使用ffmpeg拆分视频
        split_video_ffmpeg(self.file_path, scene_ranges, output_file_template=output_file_template)

        # 保存场景统计信息到CSV文件
        STATS_FILE_PATH = self.csv_path
        scene_manager.stats_manager.save_to_csv(csv_file=STATS_FILE_PATH)

        # 从生成的视频中抓取代表帧
        video_path = output_directory
        self.capture_representative_frame(video_path)

        # 发射处理完成信号
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
        finished = pyqtSignal()
        # 添加两字典，用于跟踪用户选择的图像和视频路径和相应复选框状态
        self.selected_images = {}
        self.selected_videos = {}
        """
        By ensuring that the self.Integrate.clicked.connect(self.integrate_videos) line comes after the creation of self.selected_videos,
        you make sure that the dictionary is available when self.integrate_videos is called.
        """
        self.Integrate.clicked.connect(self.integrate_videos)


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
                        layout.addWidget(checkbox, i // 5, i % 5)

                # 将 widget 设置为滚动区域的子部件
                self.ScrollArea.setWidget(widget)

        except Exception as e:
            print(f"Error in ImagePresenter: {e}")

    def integrate_videos(self):
        self.Integrate.setEnabled(False)

        # 获取所有被选中的视频路径
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

                # 提示用户整合完成
                QMessageBox.information(self, "整合完成", "视频整合完成", QMessageBox.Ok)
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

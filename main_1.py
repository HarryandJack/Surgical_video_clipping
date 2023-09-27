import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox,QProgressBar ,QLabel, QSpacerItem, QSizePolicy, QDialog, QVBoxLayout, QWidget
from demo import Ui_MainWindow
import os
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QThread
from PyQt5.uic import loadUi
import subprocess

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
        command = [
            'ffmpeg',
            '-i', self.input_file,
            '-ss', str(self.start_time),
            '-to', str(self.end_time),
            self.output_file
        ]

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        while True:
            output = process.stderr.readline()
            if process.poll() is not None:
                break
            if output:
                progress = self.parse_progress(output)
                # 在 VideoClipper 类中定义的信号 progressChanged 被发射的语句。
                self.progressChanged.emit(progress)

        process.communicate()
        self.finished.emit()

    def parse_progress(self, output):
        # 在这里解析 FFmpeg 输出的进度信息
        # 这可能需要根据实际情况进行定制
        pass

class ImagePresent(QThread):
    finished = pyqtSignal()
    progressChanged = pyqtSignal(int)

# 一个名为 VideoProcessor 的类，继承自 QThread，用于处理视频
class VideoProcessor(QThread):
    finished = pyqtSignal()
    progressChanged = pyqtSignal(int)
    def __init__(self, file_path):
        #调用了父类 QThread 的构造函数，初始化了线程
        super().__init__()
        #构造函数接受一个参数 file_path，表示要处理的视频文件的路径
        self.file_path = file_path

    def run(self):
        #使用 OpenCV 的 cv2.VideoCapture() 方法打开视频文件
        cap = cv2.VideoCapture(self.file_path)
        #设置一个文件夹路径，用于存储处理后的图片
        folder_path = 'images'

        # 如果文件夹存在，则删除其中的所有文件
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                os.remove(file_path)
        else:
            os.makedirs(folder_path)

        # 初始化一个变量 prev_frame，用于存储前一帧的灰度图像
        prev_frame = None

        # 初始化一个变量 frame_count，用于记录处理的帧数
        frame_count = 0

        # 进入一个无限循环，开始处理视频帧
        while True:
            # 读取视频的一帧，ret 表示是否成功读取，frame 是读取到的帧
            ret, frame = cap.read()
            if not ret:
                break

            # 帧数加一
            frame_count += 1

            # 每处理 100 帧执行一次以下操作
            if frame_count % 100 == 0:

                # 将彩色帧转换为灰度图像。
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # 如果前一帧为空（第一帧），将当前帧赋给 prev_frame 并继续下一轮循环
                if prev_frame is None:
                    prev_frame = gray_frame
                    continue

                # 计算前一帧和当前帧的差异
                diff = cv2.absdiff(prev_frame, gray_frame)

                # 设置一个阈值
                threshold = 100

                # 对差异进行二值化处理
                _, thresholded_diff = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

                # 如果二值化后的差异中非零像素超过 1000 个
                if cv2.countNonZero(thresholded_diff) > 1000:
                    self.progressChanged.emit(frame_count)
                    #构造保存图片的路径。
                    image_path = f"images/frame_{frame_count}.jpg"

                    #将当前帧保存为图片
                    cv2.imwrite(image_path, frame)
                    print(f"保存图片：{image_path}")

                # 更新前一帧
                prev_frame = gray_frame
        # 释放视频对象
        cap.release()
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

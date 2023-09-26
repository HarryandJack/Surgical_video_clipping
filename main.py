import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from demo import Ui_MainWindow
import os
from PyQt5.QtCore import QThread, pyqtSignal
import subprocess

class VideoClipper(QThread):
    finished = pyqtSignal()
    progressChanged = pyqtSignal(int)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        start_time = 240
        end_time = 260

        # 使用FFmpeg进行视频剪辑
        command = [
            'ffmpeg',
            '-i', self.file_path,
            '-ss', str(start_time),
            '-to', str(end_time),
            'output_video.avi'
        ]

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        while True:
            output = process.stderr.readline()
            if process.poll() is not None:
                break
            if output:
                progress = self.parse_progress(output)
                self.progressChanged.emit(progress)

        process.communicate()
        self.finished.emit()
class VideoProcessor(QThread):
    finished = pyqtSignal()

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        cap = cv2.VideoCapture(self.file_path)

        folder_path = 'images'

        # 如果文件夹存在，则删除其中的所有文件
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                os.remove(file_path)
        else:
            os.makedirs(folder_path)

        prev_frame = None
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % 100 == 0:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if prev_frame is None:
                    prev_frame = gray_frame
                    continue

                diff = cv2.absdiff(prev_frame, gray_frame)

                threshold = 100
                _, thresholded_diff = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

                if cv2.countNonZero(thresholded_diff) > 1000:
                    image_path = f"images/frame_{frame_count}.jpg"
                    cv2.imwrite(image_path, frame)
                    print(f"保存图片：{image_path}")
                prev_frame = gray_frame

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

        self.file_path, _ = QFileDialog.getOpenFileName(None, "Open Video File", "", "Video Files (*.mp4 *.avi);;All Files (*)", options=options)

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
        if self.file_path:
            self.video_clipper = VideoClipper(self.file_path)
            self.video_clipper.finished.connect(self.clip_finished)
            self.video_clipper.start()

        else:
            QMessageBox.warning(None, "剪辑失败", "未选择文件或文件无效", QMessageBox.Ok)

    def clip_finished(self):
        QMessageBox.information(None, "剪辑完成", "视频剪辑完成", QMessageBox.Ok)


    def remove_similar_frames(self):
        if self.selected_image_path and os.path.exists(self.selected_image_path):
            # TODO: 在这里实现删除相似帧的逻辑
            pass
        else:
            QMessageBox.warning(None, "操作失败", "未选择图片或图片不存在", QMessageBox.Ok)

    def export_video(self):
        if self.selected_image_path and os.path.exists(self.selected_image_path):
            # TODO: 在这里实现导出视频的逻辑
            pass
        else:
            QMessageBox.warning(None, "操作失败", "未选择图片或图片不存在", QMessageBox.Ok)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MyWindow = MyWindow()
    MyWindow.show()
    sys.exit(app.exec_())

from PyQt5.QtCore import QThread, pyqtSignal
from Tools.__init__ import open_video, ContentDetector, SceneManager
from Tools.stats_manager import StatsManager
from Tools.video_splitter import split_video_ffmpeg
import os
import cv2

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
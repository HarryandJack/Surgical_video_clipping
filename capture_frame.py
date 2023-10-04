import os
import cv2
def capture_representative_frame(video_path):
    if not os.path.exists('video_path/images'):
        os.makedirs('video_path/images')

    # 遍历 video 文件夹中的所有视频
    for filename in os.listdir('video'):
        if filename.endswith('.mp4') or filename.endswith('.avi'):  # 确保是视频文件
            video_path = os.path.join('video', filename)

            # 使用 OpenCV 打开视频文件
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"无法打开视频文件 {filename}")
                continue

            # 获取视频的总帧数
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 获取代表性的一帧（这里是取第一帧）
            ret, frame = cap.read()
            if not ret:
                print(f"无法读取视频帧 {filename}")
                continue

            # 保存代表性帧到 images 文件夹中
            image_path = os.path.join('video/images', f'{filename}_frame.jpg')
            cv2.imwrite(image_path, frame)

            # 释放视频对象
            cap.release()
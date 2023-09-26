import cv2
def clip_video(video_path,save_path,img_time):
    """
    对视频任意时间段进行剪切
    :return:
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('video is not opened')
    else:
        success, frame = cap.read()
        f_shape = frame.shape
        f_height = f_shape[0]  # 原视频图片的高度
        f_width = f_shape[1]
        fps = cap.get(5)  # 帧速率
        frame_number = cap.get(7)  # 视频文件的帧数
        duration = frame_number / fps  # 视频总帧数/帧速率 是时间/秒【总共有多少秒的视频时间】
        print('请注意视频的总时间长度为 %s 秒' % str(duration))
        # AVI格式编码输出 XVID
        four_cc = cv2.VideoWriter_fourcc(*'XVID')
        # 确定保存格式
        video_writer = cv2.VideoWriter(save_path, four_cc, fps, (int(f_width), int(f_height)))
        num = 0
        while True:
            success, frame = cap.read()
            # num/fps 结果是该帧出现的时间（单位 s)*10 是为了提高提取视频的精确到，以精确到0.1s
            if int(10*num/fps) in img_time:
                if success:
                    video_writer.write(frame)
                else:
                    break
            num += 1
            if num > frame_number:
                break
        cap.release()
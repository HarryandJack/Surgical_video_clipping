import os
from scenedetect import open_video, ContentDetector, SceneManager, StatsManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.video_splitter import split_video_ffmpeg

def run(file_path):
    video = open_video(file_path)

    scene_manager = SceneManager(stats_manager=StatsManager())

    content_detector = ContentDetector()

    scene_manager.add_detector(content_detector)

    scene_manager.detect_scenes(video=video)

    scene_list = scene_manager.get_scene_list()

    # Extract start and end frames for each scene
    scene_ranges = [(start_frame, end_frame) for start_frame, end_frame in scene_list]

    output_directory = os.path.join(os.getcwd(), 'images')  # Output directory: current working directory/images

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    output_file_template = os.path.join(output_directory, '$VIDEO_NAME-Scene-$SCENE_NUMBER.mp4')

    split_video_ffmpeg(file_path, scene_ranges, output_file_template=output_file_template)

run(r'D:\test-2.mp4')

"""``scenedetect.scene_manager`` 模块

该模块实现了 :class:`SceneManager`，它协调在视频的帧上运行一个 :mod:`SceneDetector <scenedetect.detectors>`。
视频解码在单独的线程中进行以提高性能。

该模块还包含其他辅助函数（例如 :func:`save_images`），可用于处理生成的场景列表。

===============================================================
用法
===============================================================

以下示例展示了如何基本使用 :class:`SceneManager`：

.. code:: python

    from scenedetect import open_video, SceneManager, ContentDetector
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    # 检测视频中从当前位置到结束的所有场景。
    scene_manager.detect_scenes(video)
    # `get_scene_list` 返回每个找到的场景的开始/结束时间码对的列表。
    scenes = scene_manager.get_scene_list()

还可以在每个检测到的场景上调用一个可选的回调函数，例如：

.. code:: python

    from scenedetect import open_video, SceneManager, ContentDetector

    # 在每次检测到新场景时调用的回调函数。
    def on_new_scene(frame_img: numpy.ndarray, frame_num: int):
        print("在帧 %d 处发现新场景。" % frame_num)

    video = open_video(test_video_file)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    scene_manager.detect_scenes(video=video, callback=on_new_scene)

要使用 `SceneManager` 与网络摄像头/设备或现有的 `cv2.VideoCapture` 设备，可以使用
:class:`VideoCaptureAdapter <scenedetect.backends.opencv.VideoCaptureAdapter>` 而不是
`open_video`。

=======================================================================
存储每帧的统计信息
=======================================================================

`SceneManager` 可以使用可选的 :class:`StatsManager <scenedetect.stats_manager.StatsManager>` 将帧统计信息保存到磁盘：

.. code:: python

    from scenedetect import open_video, ContentDetector, SceneManager, StatsManager
    video = open_video(test_video_file)
    scene_manager = SceneManager(stats_manager=StatsManager())
    scene_manager.add_detector(ContentDetector())
    scene_manager.detect_scenes(video=video)
    scene_list = scene_manager.get_scene_list()
    print_scenes(scene_list=scene_list)
    # 将每帧的统计信息保存到磁盘。
    scene_manager.stats_manager.save_to_csv(csv_file=STATS_FILE_PATH)

可以使用统计文件来找到某些输入的更好阈值，或对视频进行统计分析。
"""

import csv
from enum import Enum
from typing import Iterable, List, Tuple, Optional, Dict, Callable, Union, TextIO
import threading
import queue
import logging
import math
import sys

import cv2
import numpy as np
from scenedetect._thirdparty.simpletable import (SimpleTableCell, SimpleTableImage, SimpleTableRow,
                                                 SimpleTable, HTMLPage)

from scenedetect.platform import (tqdm, get_and_create_path, get_cv2_imwrite_params, Template)
from Tools.frame_timecode import FrameTimecode
from Tools.video_stream import VideoStream
from Tools.scene_detector import SceneDetector, SparseSceneDetector
from Tools.stats_manager import StatsManager, FrameMetricRegistered

logger = logging.getLogger('pyscenedetect')

# TODO: This value can and should be tuned for performance improvements as much as possible,
# until accuracy falls, on a large enough dataset. This has yet to be done, but the current
# value doesn't seem to have caused any issues at least.
DEFAULT_MIN_WIDTH: int = 256
"""The default minimum width a frame will be downscaled to when calculating a downscale factor."""

MAX_FRAME_QUEUE_LENGTH: int = 4
"""Maximum number of decoded frames which can be buffered while waiting to be processed."""

PROGRESS_BAR_DESCRIPTION = 'Detected: %d | Progress'
"""Template to use for progress bar."""


class Interpolation(Enum):
    """Interpolation method used for image resizing. Based on constants defined in OpenCV."""
    NEAREST = cv2.INTER_NEAREST
    """Nearest neighbor interpolation."""
    LINEAR = cv2.INTER_LINEAR
    """Bilinear interpolation."""
    CUBIC = cv2.INTER_CUBIC
    """Bicubic interpolation."""
    AREA = cv2.INTER_AREA
    """Pixel area relation resampling. Provides moire'-free downscaling."""
    LANCZOS4 = cv2.INTER_LANCZOS4
    """Lanczos interpolation over 8x8 neighborhood."""


def compute_downscale_factor(frame_width: int, effective_width: int = DEFAULT_MIN_WIDTH) -> int:
    """根据视频的分辨率（目前仅考虑像素宽度），获取最佳的默认缩小因子。

    得到的视频有效宽度将在 frame_width 和 1.5 * frame_width 像素之间（例如，如果 frame_width 为 200，那么有效宽度的范围将在 200 到 300 之间）。

    参数:
    frame_width: 视频帧的实际宽度，以像素为单位。
    effective_width: 所需的最小宽度，以像素为单位。

    返回:
    int: 用于至少达到目标 effective_width 的默认缩小因子。
    """
    assert not (frame_width < 1 or effective_width < 1)
    if frame_width < effective_width:
        return 1
    return frame_width // effective_width


def get_scenes_from_cuts(
    cut_list: Iterable[FrameTimecode],
    start_pos: Union[int, FrameTimecode],
    end_pos: Union[int, FrameTimecode],
    base_timecode: Optional[FrameTimecode] = None,
) -> List[Tuple[FrameTimecode, FrameTimecode]]:
    """Returns a list of tuples of start/end FrameTimecodes for each scene based on a
    list of detected scene cuts/breaks.

    This function is called when using the :meth:`SceneManager.get_scene_list` method.
    The scene list is generated from a cutting list (:meth:`SceneManager.get_cut_list`),
    noting that each scene is contiguous, starting from the first to last frame of the input.
    If `cut_list` is empty, the resulting scene will span from `start_pos` to `end_pos`.

    Arguments:
        cut_list: List of FrameTimecode objects where scene cuts/breaks occur.
        base_timecode: The base_timecode of which all FrameTimecodes in the cut_list are based on.
        num_frames: The number of frames, or FrameTimecode representing duration, of the video that
            was processed (used to generate last scene's end time).
        start_frame: The start frame or FrameTimecode of the cut list. Used to generate the first
            scene's start time.
            base_timecode: [DEPRECATED] DO NOT USE. For backwards compatibility only.
    Returns:
        List of tuples in the form (start_time, end_time), where both start_time and
        end_time are FrameTimecode objects representing the exact time/frame where each
        scene occupies based on the input cut_list.
    """
    # TODO(v0.7): Use the warnings module to turn this into a warning.
    if base_timecode is not None:
        logger.error('`base_timecode` argument is deprecated has no effect.')

    # Scene list, where scenes are tuples of (Start FrameTimecode, End FrameTimecode).
    scene_list = []
    if not cut_list:
        scene_list.append((start_pos, end_pos))
        return scene_list
    # Initialize last_cut to the first frame we processed,as it will be
    # the start timecode for the first scene in the list.
    last_cut = start_pos
    for cut in cut_list:
        scene_list.append((last_cut, cut))
        last_cut = cut
    # Last scene is from last cut to end of video.
    scene_list.append((last_cut, end_pos))

    return scene_list


def write_scene_list(output_csv_file: TextIO,
                     scene_list: Iterable[Tuple[FrameTimecode, FrameTimecode]],
                     include_cut_list: bool = True,
                     cut_list: Optional[Iterable[FrameTimecode]] = None) -> None:
    """Writes the given list of scenes to an output file handle in CSV format.

    Arguments:
        output_csv_file: Handle to open file in write mode.
        scene_list: List of pairs of FrameTimecodes denoting each scene's start/end FrameTimecode.
        include_cut_list: Bool indicating if the first row should include the timecodes where
            each scene starts. Should be set to False if RFC 4180 compliant CSV output is required.
        cut_list: Optional list of FrameTimecode objects denoting the cut list (i.e. the frames
            in the video that need to be split to generate individual scenes). If not specified,
            the cut list is generated using the start times of each scene following the first one.
    """
    csv_writer = csv.writer(output_csv_file, lineterminator='\n')
    # If required, output the cutting list as the first row (i.e. before the header row).
    if include_cut_list:
        csv_writer.writerow(
            ["Timecode List:"] +
            cut_list if cut_list else [start.get_timecode() for start, _ in scene_list[1:]])
    csv_writer.writerow([
        "Scene Number", "Start Frame", "Start Timecode", "Start Time (seconds)", "End Frame",
        "End Timecode", "End Time (seconds)", "Length (frames)", "Length (timecode)",
        "Length (seconds)"
    ])
    for i, (start, end) in enumerate(scene_list):
        duration = end - start
        csv_writer.writerow([
            '%d' % (i + 1),
            '%d' % (start.get_frames() + 1),
            start.get_timecode(),
            '%.3f' % start.get_seconds(),
            '%d' % end.get_frames(),
            end.get_timecode(),
            '%.3f' % end.get_seconds(),
            '%d' % duration.get_frames(),
            duration.get_timecode(),
            '%.3f' % duration.get_seconds()
        ])


def write_scene_list_html(output_html_filename,
                          scene_list,
                          cut_list=None,
                          css=None,
                          css_class='mytable',
                          image_filenames=None,
                          image_width=None,
                          image_height=None):
    """Writes the given list of scenes to an output file handle in html format.

    Arguments:
        output_html_filename: filename of output html file
        scene_list: List of pairs of FrameTimecodes denoting each scene's start/end FrameTimecode.
        cut_list: Optional list of FrameTimecode objects denoting the cut list (i.e. the frames
            in the video that need to be split to generate individual scenes). If not passed,
            the start times of each scene (besides the 0th scene) is used instead.
        css: String containing all the css information for the resulting html page.
        css_class: String containing the named css class
        image_filenames: dict where key i contains a list with n elements (filenames of
            the n saved images from that scene)
        image_width: Optional desired width of images in table in pixels
        image_height: Optional desired height of images in table in pixels
    """
    if not css:
        css = """
        table.mytable {
            font-family: times;
            font-size:12px;
            color:#000000;
            border-width: 1px;
            border-color: #eeeeee;
            border-collapse: collapse;
            background-color: #ffffff;
            width=100%;
            max-width:550px;
            table-layout:fixed;
        }
        table.mytable th {
            border-width: 1px;
            padding: 8px;
            border-style: solid;
            border-color: #eeeeee;
            background-color: #e6eed6;
            color:#000000;
        }
        table.mytable td {
            border-width: 1px;
            padding: 8px;
            border-style: solid;
            border-color: #eeeeee;
        }
        #code {
            display:inline;
            font-family: courier;
            color: #3d9400;
        }
        #string {
            display:inline;
            font-weight: bold;
        }
        """

    # Output Timecode list
    timecode_table = SimpleTable(
        [["Timecode List:"] +
         (cut_list if cut_list else [start.get_timecode() for start, _ in scene_list[1:]])],
        css_class=css_class)

    # Output list of scenes
    header_row = [
        "Scene Number", "Start Frame", "Start Timecode", "Start Time (seconds)", "End Frame",
        "End Timecode", "End Time (seconds)", "Length (frames)", "Length (timecode)",
        "Length (seconds)"
    ]
    for i, (start, end) in enumerate(scene_list):
        duration = end - start

        row = SimpleTableRow([
            '%d' % (i + 1),
            '%d' % (start.get_frames() + 1),
            start.get_timecode(),
            '%.3f' % start.get_seconds(),
            '%d' % end.get_frames(),
            end.get_timecode(),
            '%.3f' % end.get_seconds(),
            '%d' % duration.get_frames(),
            duration.get_timecode(),
            '%.3f' % duration.get_seconds()
        ])

        if image_filenames:
            for image in image_filenames[i]:
                row.add_cell(
                    SimpleTableCell(
                        SimpleTableImage(image, width=image_width, height=image_height)))

        if i == 0:
            scene_table = SimpleTable(rows=[row], header_row=header_row, css_class=css_class)
        else:
            scene_table.add_row(row=row)

    # Write html file
    page = HTMLPage()
    page.add_table(timecode_table)
    page.add_table(scene_table)
    page.css = css
    page.save(output_html_filename)


#
# TODO(v1.0): Refactor to take a SceneList object; consider moving this and save scene list
# to a better spot, or just move them to scene_list.py.
#
def save_images(scene_list: List[Tuple[FrameTimecode, FrameTimecode]],
                video: VideoStream,
                num_images: int = 3,
                frame_margin: int = 1,
                image_extension: str = 'jpg',
                encoder_param: int = 95,
                image_name_template: str = '$VIDEO_NAME-Scene-$SCENE_NUMBER-$IMAGE_NUMBER',
                output_dir: Optional[str] = None,
                show_progress: Optional[bool] = False,
                scale: Optional[float] = None,
                height: Optional[int] = None,
                width: Optional[int] = None,
                interpolation: Interpolation = Interpolation.CUBIC,
                video_manager=None) -> Dict[int, List[str]]:
    """根据给定的场景列表和相关的视频/帧源，保存每个场景中的一定数量的图像。

参数:
    scene_list: 从调用SceneManager的detect_scenes()方法返回的场景列表（由FrameTimecode对象成对组成）。
    video: 与场景列表相对应的VideoStream对象。
        请注意，视频将会被关闭/重新打开并在其中进行搜索。
    num_images: 每个场景要生成的图像数量。最小值为1。
    frame_margin: 在每个场景的开头和结尾周围填充的帧数（例如，将第一个/最后一个图像移动到场景中N帧）。
        可以设置为0，但会导致某些视频文件无法提取最后一帧。
    image_extension: 要保存的图像类型（必须是'jpg'、'png'或'webp'之一）。
    encoder_param: 压缩效率/质量参数，根据图像类型而定:
        'jpg' / 'webp': 0-100的质量，值越高质量越好。对于webp，100是无损的。
        'png': 从1-9的压缩等级，9获得最佳文件大小但编码速度较慢。
    image_name_template: 在创建磁盘上的图像时要使用的模板。可以使用宏$VIDEO_NAME，$SCENE_NUMBER和$IMAGE_NUMBER。
        图像扩展名将根据参数image_extension自动应用。
    output_dir: 输出图像的目录。如果未设置，则将在当前工作目录中创建输出目录。
    show_progress: 如果为True，则在安装了tqdm时显示进度条。
    scale: 保存图像的可选缩放因子。缩放因子为1将不会进行缩放。值<1会导致较小的保存图像，而值>1会导致比原始图像大的图像。
        如果指定了高度或宽度值，则将忽略此值。
    height: 保存图像的可选高度值。同时指定高度和宽度将会将图像调整为确切的大小，而不考虑纵横比。
        仅指定高度将会将图像调整为在保持纵横比的情况下拥有指定高度像素数。
    width: 保存图像的可选宽度值。同时指定宽度和高度将会将图像调整为确切的大小，而不考虑纵横比。
        仅指定宽度将会将图像调整为在保持纵横比的情况下拥有指定宽度像素数。
    interpolation: 调整图像大小时要使用的插值类型。
    video_manager: [已弃用] 请勿使用。仅用于向后兼容性。

返回:
    字典，格式为 { 场景编号 : [图像路径] }，其中场景编号是scene_list中的场景编号（从1开始），图像路径是新保存/创建的图像的路径列表。

抛出:
    ValueError: 如果任何参数无效或超出范围（例如，如果num_images为负数）。
"""

    # TODO(v0.7): Add DeprecationWarning that `video_manager` will be removed in v0.8.
    if video_manager is not None:
        logger.error('`video_manager` argument is deprecated, use `video` instead.')
        video = video_manager

    if not scene_list:
        return {}
    if num_images <= 0 or frame_margin < 0:
        raise ValueError()

    # TODO: Validate that encoder_param is within the proper range.
    # Should be between 0 and 100 (inclusive) for jpg/webp, and 1-9 for png.
    imwrite_param = [get_cv2_imwrite_params()[image_extension], encoder_param
                    ] if encoder_param is not None else []

    video.reset()

    # Setup flags and init progress bar if available.
    completed = True
    logger.info('Generating output images (%d per scene)...', num_images)
    progress_bar = None
    if show_progress:
        progress_bar = tqdm(total=len(scene_list) * num_images, unit='images', dynamic_ncols=True)

    filename_template = Template(image_name_template)

    scene_num_format = '%0'
    scene_num_format += str(max(3, math.floor(math.log(len(scene_list), 10)) + 1)) + 'd'
    image_num_format = '%0'
    image_num_format += str(math.floor(math.log(num_images, 10)) + 2) + 'd'

    framerate = scene_list[0][0].framerate

    # TODO(v1.0): Split up into multiple sub-expressions so auto-formatter works correctly.
    timecode_list = [
        [
            FrameTimecode(int(f), fps=framerate) for f in [
                                                                                               # middle frames
                a[len(a) // 2] if (0 < j < num_images - 1) or num_images == 1

                                                                                               # first frame
                else min(a[0] + frame_margin, a[-1]) if j == 0

                                                                                               # last frame
                else max(a[-1] - frame_margin, a[0])

                                                                                               # for each evenly-split array of frames in the scene list
                for j, a in enumerate(np.array_split(r, num_images))
            ]
        ] for i, r in enumerate([
                                                                                               # pad ranges to number of images
            r if 1 + r[-1] - r[0] >= num_images else list(r) + [r[-1]] * (num_images - len(r))
                                                                                               # create range of frames in scene
            for r in (
                range(
                    start.get_frames(),
                    start.get_frames() + max(
                        1,                                                                     # guard against zero length scenes
                        end.get_frames() - start.get_frames()))
                                                                                               # for each scene in scene list
                for start, end in scene_list)
        ])
    ]

    image_filenames = {i: [] for i in range(len(timecode_list))}
    aspect_ratio = video.aspect_ratio
    if abs(aspect_ratio - 1.0) < 0.01:
        aspect_ratio = None

    logger.debug('Writing images with template %s', filename_template.template)
    for i, scene_timecodes in enumerate(timecode_list):
        for j, image_timecode in enumerate(scene_timecodes):
            video.seek(image_timecode)
            frame_im = video.read()
            if frame_im is not None:
                # TODO: Allow NUM to be a valid suffix in addition to NUMBER.
                file_path = '%s.%s' % (filename_template.safe_substitute(
                    VIDEO_NAME=video.name,
                    SCENE_NUMBER=scene_num_format % (i + 1),
                    IMAGE_NUMBER=image_num_format % (j + 1),
                    FRAME_NUMBER=image_timecode.get_frames()), image_extension)
                image_filenames[i].append(file_path)
                # TODO(0.6.3): Combine this resize with the ones below.
                if aspect_ratio is not None:
                    frame_im = cv2.resize(
                        frame_im, (0, 0),
                        fx=aspect_ratio,
                        fy=1.0,
                        interpolation=interpolation.value)
                frame_height = frame_im.shape[0]
                frame_width = frame_im.shape[1]

                # Figure out what kind of resizing needs to be done
                if height or width:
                    if height and not width:
                        factor = height / float(frame_height)
                        width = int(factor * frame_width)
                    if width and not height:
                        factor = width / float(frame_width)
                        height = int(factor * frame_height)
                    assert height > 0 and width > 0
                    frame_im = cv2.resize(
                        frame_im, (width, height), interpolation=interpolation.value)
                elif scale:
                    frame_im = cv2.resize(
                        frame_im, (0, 0), fx=scale, fy=scale, interpolation=interpolation.value)

                cv2.imwrite(get_and_create_path(file_path, output_dir), frame_im, imwrite_param)
            else:
                completed = False
                break
            if progress_bar is not None:
                progress_bar.update(1)

    if progress_bar is not None:
        progress_bar.close()

    if not completed:
        logger.error('Could not generate all output images.')

    return image_filenames




class SceneManager:
    """SceneManager 管理视频中的场景检测(:meth:`detect_scenes`)，使用检测器(:meth:`add_detector`)。
    视频解码在后台线程中进行。
    """

    def __init__(
        self,
        stats_manager: Optional[StatsManager] = None,
    ):
        """
        参数:
            stats_manager: 要绑定到此 `SceneManager` 的 :class:`StatsManager`。
                可以通过生成的对象的 `stats_manager` 属性来访问以保存到磁盘。
        """
        self._cutting_list = []  # 切割列表
        self._event_list = []  # 事件列表
        self._detector_list = []  # 检测器列表
        self._sparse_detector_list = []  # 稀疏检测器列表

        # TODO(v1.0): 这个类应该拥有一个 StatsManager，而不是接受一个可选的 StatsManager。
        # 通过 `stats_manager` 属性访问结果对象，以便保存到磁盘。
        self._stats_manager: Optional[StatsManager] = stats_manager

        # TODO(v1.0): 这个类也应该拥有一个 VideoStream，而不是将其传递给 detect_scenes 方法。
        # 如果需要连接，可以实现为一个通用的 VideoStream 包装器。

        # 视频第一次传递到 detect_scenes 的位置。
        self._start_pos: FrameTimecode = None
        # detect_scenes 处理的最后一帧的视频位置。
        self._last_pos: FrameTimecode = None
        self._base_timecode: Optional[FrameTimecode] = None
        self._downscale: int = 1
        self._auto_downscale: bool = True
        # 缩小时要使用的插值方法。默认为线性插值，以在质量和性能之间取得良好的平衡。
        self._interpolation: Interpolation = Interpolation.LINEAR
        # 布尔值，指示到目前为止我们只看到 EventType.CUT 事件。
        self._only_cuts: bool = True
        # 在解码线程中发生异常时设置。
        self._exception_info = None
        self._stop = threading.Event()

        self._frame_buffer = []  # 帧缓冲区
        self._frame_buffer_size = 0  # 帧缓冲区大小

    @property
    def interpolation(self) -> Interpolation:
        """Interpolation method to use when downscaling frames. Must be one of cv2.INTER_*."""
        return self._interpolation

    @interpolation.setter
    def interpolation(self, value: Interpolation):
        self._interpolation = value

    @property
    def stats_manager(self) -> Optional[StatsManager]:
        """Getter for the StatsManager associated with this SceneManager, if any."""
        return self._stats_manager

    @property
    def downscale(self) -> int:
        """Factor to downscale each frame by. Will always be >= 1, where 1
        indicates no scaling. Will be ignored if auto_downscale=True."""
        return self._downscale

    @downscale.setter
    def downscale(self, value: int):
        """Set to 1 for no downscaling, 2 for 2x downscaling, 3 for 3x, etc..."""
        if value < 1:
            raise ValueError("Downscale factor must be a positive integer >= 1!")
        if self.auto_downscale:
            logger.warning("Downscale factor will be ignored because auto_downscale=True!")
        if value is not None and not isinstance(value, int):
            logger.warning("Downscale factor will be truncated to integer!")
            value = int(value)
        self._downscale = value

    @property
    def auto_downscale(self) -> bool:
        """If set to True, will automatically downscale based on video frame size.

        Overrides `downscale` if set."""
        return self._auto_downscale

    @auto_downscale.setter
    def auto_downscale(self, value: bool):
        self._auto_downscale = value

    def add_detector(self, detector: SceneDetector) -> None:
        """
    将 SceneDetector（例如 ContentDetector、ThresholdDetector）添加/注册到在调用 detect_scenes 时运行。
    SceneManager 拥有检测器对象，因此可以传递临时对象。

    参数:
    detector (SceneDetector): 要添加到 SceneManager 的场景检测器。
        """

        if self._stats_manager is None and detector.stats_manager_required():
            # Make sure the lists are empty so that the detectors don't get
            # out of sync (require an explicit statsmanager instead)
            assert not self._detector_list and not self._sparse_detector_list
            self._stats_manager = StatsManager()

        detector.stats_manager = self._stats_manager
        if self._stats_manager is not None:
            try:
                self._stats_manager.register_metrics(detector.get_metrics())
            except FrameMetricRegistered:
                # Allow multiple detection algorithms of the same type to be added
                # by suppressing any FrameMetricRegistered exceptions due to attempts
                # to re-register the same frame metric keys.
                # TODO(#334): Fix this, this should not be part of regular control flow.
                pass

        if not issubclass(type(detector), SparseSceneDetector):
            self._detector_list.append(detector)
        else:
            self._sparse_detector_list.append(detector)

        self._frame_buffer_size = max(detector.event_buffer_length, self._frame_buffer_size)

    def get_num_detectors(self) -> int:
        """Get number of registered scene detectors added via add_detector. """
        return len(self._detector_list)

    def clear(self) -> None:
        """
    清除所有的切割/场景，并重置 SceneManager 的位置。

    生成的任何统计信息仍然保存在传递给 SceneManager 构造函数的 StatsManager 对象中，
    因此，后续对 detect_scenes 的调用，使用相同的帧源定位回原始时间（或视频的开始），
    将使用在先前对 detect_scenes 的调用中计算并保存的缓存帧指标。
        """

        self._cutting_list.clear()
        self._event_list.clear()
        self._last_pos = None
        self._start_pos = None
        self.clear_detectors()

    def clear_detectors(self) -> None:
        """Remove all scene detectors added to the SceneManager via add_detector(). """
        self._detector_list.clear()
        self._sparse_detector_list.clear()

    def get_scene_list(self,
                       base_timecode: Optional[FrameTimecode] = None,
                       start_in_scene: bool = False) -> List[Tuple[FrameTimecode, FrameTimecode]]:
        """
    返回一个包含每个检测到的场景的起始/结束帧时间码的列表。

    参数:
        - base_timecode: [已弃用] 请勿使用。用于向后兼容。
        - start_in_scene: 假设视频开始于一个场景。这意味着当使用 `ContentDetector` 进行快速切割检测时，
        如果未发现任何切割，则生成的场景列表将包含一个跨足整个视频的单个场景（而不是没有场景）。
        当使用 `ThresholdDetector` 进行淡入淡出检测时，视频的起始部分将始终包括在内，
        直到检测到第一个淡出事件。

    返回:
        一个由元组组成的列表，每个元组的形式为 (start_time, end_time)，
        其中 start_time 和 end_time 都是 FrameTimecode 对象，表示视频中每个检测到的场景开始和结束的确切时间/帧。
        """
        # TODO(v0.7): Replace with DeprecationWarning that `base_timecode` will be removed in v0.8.
        if base_timecode is not None:
            logger.error('`base_timecode` argument is deprecated and has no effect.')
        if self._base_timecode is None:
            return []
        cut_list = self._get_cutting_list()
        scene_list = get_scenes_from_cuts(
            cut_list=cut_list, start_pos=self._start_pos, end_pos=self._last_pos + 1)
        # If we didn't actually detect any cuts, make sure the resulting scene_list is empty
        # unless start_in_scene is True.
        if not cut_list and not start_in_scene:
            scene_list = []
        return sorted(self._get_event_list() + scene_list)

    def _get_cutting_list(self) -> List[int]:
        """Return a sorted list of unique frame numbers of any detected scene cuts."""
        if not self._cutting_list:
            return []
        assert self._base_timecode is not None
        # Ensure all cuts are unique by using a set to remove all duplicates.
        return [self._base_timecode + cut for cut in sorted(set(self._cutting_list))]

    def _get_event_list(self) -> List[Tuple[FrameTimecode, FrameTimecode]]:
        if not self._event_list:
            return []
        assert self._base_timecode is not None
        return [(self._base_timecode + start, self._base_timecode + end)
                for start, end in self._event_list]

    def _process_frame(self,
                       frame_num: int,
                       frame_im: np.ndarray,
                       callback: Optional[Callable[[np.ndarray, int], None]] = None) -> bool:
        """Add any cuts detected with the current frame to the cutting list. Returns True if any new
        cuts were detected, False otherwise."""
        new_cuts = False
        # TODO(#283): This breaks with AdaptiveDetector as cuts differ from the frame number
        # being processed. Allow detectors to specify the max frame lookahead they require
        # (i.e. any event will never be more than N frames behind the current one).
        self._frame_buffer.append(frame_im)
        # frame_buffer[-1] is current frame, -2 is one behind, etc
        # so index based on cut frame should be [event_frame - (frame_num + 1)]
        self._frame_buffer = self._frame_buffer[-(self._frame_buffer_size + 1):]
        for detector in self._detector_list:
            cuts = detector.process_frame(frame_num, frame_im)
            self._cutting_list += cuts
            new_cuts = True if cuts else False
            if callback:
                for cut_frame_num in cuts:
                    buffer_index = cut_frame_num - (frame_num + 1)
                    callback(self._frame_buffer[buffer_index], cut_frame_num)
        for detector in self._sparse_detector_list:
            events = detector.process_frame(frame_num, frame_im)
            self._event_list += events
            if callback:
                for event_start, _ in events:
                    buffer_index = event_start - (frame_num + 1)
                    callback(self._frame_buffer[buffer_index], event_start)
        return new_cuts

    def _post_process(self, frame_num: int) -> None:
        """Add remaining cuts to the cutting list, after processing the last frame."""
        for detector in self._detector_list:
            self._cutting_list += detector.post_process(frame_num)

    def stop(self) -> None:
        """Stop the current :meth:`detect_scenes` call, if any. Thread-safe."""
        self._stop.set()

    def detect_scenes(self,
                      video: VideoStream = None,
                      duration: Optional[FrameTimecode] = None,
                      end_time: Optional[FrameTimecode] = None,
                      frame_skip: int = 0,
                      show_progress: bool = False,
                      callback: Optional[Callable[[np.ndarray, int], None]] = None,
                      frame_source: Optional[VideoStream] = None) -> int:
        """
在给定的视频上使用添加的 SceneDetectors 执行场景检测，返回处理的帧数。
可以通过调用 :meth:`get_scene_list` 或 :meth:`get_cut_list` 获取结果。

视频解码在后台线程中执行，以允许场景检测和帧解码并行进行。
检测将持续进行，直到没有剩余帧，达到指定的持续时间或结束时间，或调用了 :meth:`stop`。

参数:
    - video: 从 `scenedetect.open_video` 获取的 VideoStream，或者直接创建一个 VideoStream 对象
      (例如 `scenedetect.backends.opencv.VideoStreamCv2`)。
    - duration: 从当前视频位置开始检测的持续时间。如果设置了 `end_time`，则不能指定。
    - end_time: 停止处理的时间点。如果设置了 `duration`，则不能指定。
    - frame_skip: 不推荐使用，除非是极高帧率的视频。要跳过的帧数（即每 N+1 帧处理一次，
      其中 N 是 frame_skip，只处理视频的 1/(N+1) 部分，以加快检测速度，但会降低准确性）。
      使用 StatsManager 时，`frame_skip` **必须**为 0（默认值）。
    - show_progress: 如果为 True，并且安装了 ``tqdm`` 模块，会显示带有进度、帧率和预计完成处理视频帧源所需的时间的进度条。
    - callback: 如果设置，则在检测到每个场景/事件后调用。
    - frame_source: [已弃用] 请勿使用。用于与早期版本兼容。

返回:
    int: 从帧源中读取和处理的帧数。
抛出:
    ValueError: 如果使用 StatsManager 对象构造了 SceneManager，则 `frame_skip` **必须**为 0（默认值）。
"""

        # TODO(v0.7): Add DeprecationWarning that `frame_source` will be removed in v0.8.
        # TODO(v0.8): Remove default value for `video`` when removing `frame_source`.
        if frame_source is not None:
            video = frame_source
        if video is None:
            raise TypeError("detect_scenes() missing 1 required positional argument: 'video'")

        if frame_skip > 0 and self.stats_manager is not None:
            raise ValueError('frame_skip must be 0 when using a StatsManager.')
        if duration is not None and end_time is not None:
            raise ValueError('duration and end_time cannot be set at the same time!')
        if duration is not None and duration < 0:
            raise ValueError('duration must be greater than or equal to 0!')
        if end_time is not None and end_time < 0:
            raise ValueError('end_time must be greater than or equal to 0!')

        self._base_timecode = video.base_timecode
        # TODO(v1.0): Fix this properly by making SceneManager create and own a StatsManager,
        # and requiring the framerate to be passed to the StatsManager the constructor.
        if self._stats_manager is not None:
            self._stats_manager._base_timecode = self._base_timecode
        start_frame_num: int = video.frame_number

        if duration is not None:
            end_time: Union[int, FrameTimecode] = duration + start_frame_num

        if end_time is not None:
            end_time: FrameTimecode = self._base_timecode + end_time

        # Can only calculate total number of frames we expect to process if the duration of
        # the video is available.
        total_frames = 0
        if video.duration is not None:
            if end_time is not None and end_time < video.duration:
                total_frames = (end_time - start_frame_num) + 1
            else:
                total_frames = (video.duration.get_frames() - start_frame_num)

        # Calculate the desired downscale factor and log the effective resolution.
        if self.auto_downscale:
            downscale_factor = compute_downscale_factor(frame_width=video.frame_size[0])
        else:
            downscale_factor = self.downscale
        if downscale_factor > 1:
            logger.info('Downscale factor set to %d, effective resolution: %d x %d',
                        downscale_factor, video.frame_size[0] // downscale_factor,
                        video.frame_size[1] // downscale_factor)

        progress_bar = None
        if show_progress:
            progress_bar = tqdm(
                total=int(total_frames),
                unit='frames',
                desc=PROGRESS_BAR_DESCRIPTION % 0,
                dynamic_ncols=True,
            )

        frame_queue = queue.Queue(MAX_FRAME_QUEUE_LENGTH)
        self._stop.clear()
        decode_thread = threading.Thread(
            target=SceneManager._decode_thread,
            args=(self, video, frame_skip, downscale_factor, end_time, frame_queue),
            daemon=True)
        decode_thread.start()
        frame_im = None

        logger.info('Detecting scenes...')
        while not self._stop.is_set():
            next_frame, position = frame_queue.get()
            if next_frame is None and position is None:
                break
            if not next_frame is None:
                frame_im = next_frame
            new_cuts = self._process_frame(position.frame_num, frame_im, callback)
            if progress_bar is not None:
                if new_cuts:
                    progress_bar.set_description(
                        PROGRESS_BAR_DESCRIPTION % len(self._cutting_list), refresh=False)
                progress_bar.update(1 + frame_skip)

        if progress_bar is not None:
            progress_bar.close()
        # Unblock any puts in the decode thread before joining. This can happen if the main
        # processing thread stops before the decode thread.
        while not frame_queue.empty():
            frame_queue.get_nowait()
        decode_thread.join()

        if self._exception_info is not None:
            raise self._exception_info[1].with_traceback(self._exception_info[2])

        self._last_pos = video.position
        self._post_process(video.position.frame_num)
        return video.frame_number - start_frame_num

    def _decode_thread(
        self,
        video: VideoStream,
        frame_skip: int,
        downscale_factor: int,
        end_time: FrameTimecode,
        out_queue: queue.Queue,
    ):
        """
        :class:`ContentDetector` 比较相邻帧之间的内容差异与设定的阈值/得分，如果超过了阈值，就会触发场景切换。

        这个检测器可以通过命令行使用 `detect-content` 命令。

        ---

        以下是 `_decode_thread` 函数的说明：

        该函数用于在单独的线程中解码视频帧，同时进行一些预处理操作（如降低分辨率）。

        参数：
            - video: VideoStream 对象，用于读取视频帧。
            - frame_skip: 帧跳过的间隔。
            - downscale_factor: 缩小因子，用于降低视频分辨率。
            - end_time: 视频结束的时间。
            - out_queue: 用于存储处理后的帧和位置信息的队列。

        注意：
            该函数会在视频解码过程中可能抛出异常，这些异常会在主线程中重新抛出以便处理。
        """
        try:
            while not self._stop.is_set():
                frame_im = None
                # We don't do any kind of locking here since the worst-case of this being wrong
                # is that we do some extra work, and this function should never mutate any data
                # (all of which should be modified under the GIL).
                # TODO(v1.0): This optimization should be removed as it is an uncommon use case and
                # greatly increases the complexity of detection algorithms using it.
                if self._is_processing_required(video.position.frame_num):
                    frame_im = video.read()
                    if frame_im is False:
                        break
                    if downscale_factor > 1:
                        frame_im = cv2.resize(
                            frame_im, (round(frame_im.shape[1] / downscale_factor),
                                       round(frame_im.shape[0] / downscale_factor)),
                            interpolation=self._interpolation.value)
                else:
                    if video.read(decode=False) is False:
                        break

                # Set the start position now that we decoded at least the first frame.
                if self._start_pos is None:
                    self._start_pos = video.position

                out_queue.put((frame_im, video.position))

                if frame_skip > 0:
                    for _ in range(frame_skip):
                        if not video.read(decode=False):
                            break
                # End time includes the presentation time of the frame, but the `position`
                # property of a VideoStream references the beginning of the frame in time.
                if end_time is not None and not (video.position + 1) < end_time:
                    break

        # If *any* exceptions occur, we re-raise them in the main thread so that the caller of
        # detect_scenes can handle it.
        except KeyboardInterrupt:
            logger.debug("Received KeyboardInterrupt.")
            self._stop.set()
        except BaseException:
            logger.critical('Fatal error: Exception raised in decode thread.')
            self._exception_info = sys.exc_info()
            self._stop.set()

        finally:
            # Handle case where start position was never set if we did not decode any frames.
            if self._start_pos is None:
                self._start_pos = video.position
            # Make sure main thread stops processing loop.
            out_queue.put((None, None))

        # pylint: enable=bare-except

    #
    # Deprecated Methods
    #

    # pylint: disable=unused-argument

    def get_cut_list(self,
                     base_timecode: Optional[FrameTimecode] = None,
                     show_warning: bool = True) -> List[FrameTimecode]:
        """[DEPRECATED] Return a list of FrameTimecodes of the detected scene changes/cuts.

        Unlike get_scene_list, the cutting list returns a list of FrameTimecodes representing
        the point in the input video where a new scene was detected, and thus the frame
        where the input should be cut/split. The cutting list, in turn, is used to generate
        the scene list, noting that each scene is contiguous starting from the first frame
        and ending at the last frame detected.

        If only sparse detectors are used (e.g. MotionDetector), this will always be empty.

        Arguments:
            base_timecode: [DEPRECATED] DO NOT USE. For backwards compatibility only.
            show_warning: If set to False, suppresses the error from being warned. In v0.7,
                this will have no effect and the error will become a Python warning.

        Returns:
            List of FrameTimecode objects denoting the points in time where a scene change
            was detected in the input video, which can also be passed to external tools
            for automated splitting of the input into individual scenes.
        """
        # TODO(v0.7): Use the warnings module to turn this into a warning.
        if show_warning:
            logger.error('`get_cut_list()` is deprecated and will be removed in a future release.')
        return self._get_cutting_list()

    def get_event_list(
            self,
            base_timecode: Optional[FrameTimecode] = None
    ) -> List[Tuple[FrameTimecode, FrameTimecode]]:
        """[DEPRECATED] DO NOT USE.

        Get a list of start/end timecodes of sparse detection events.

        Unlike get_scene_list, the event list returns a list of FrameTimecodes representing
        the point in the input video where a new scene was detected only by sparse detectors,
        otherwise it is the same.

        Arguments:
            base_timecode: [DEPRECATED] DO NOT USE. For backwards compatibility only.

        Returns:
            List of pairs of FrameTimecode objects denoting the detected scenes.
        """
        # TODO(v0.7): Use the warnings module to turn this into a warning.
        logger.error('`get_event_list()` is deprecated and will be removed in a future release.')
        return self._get_event_list()

    # pylint: enable=unused-argument

    def _is_processing_required(self, frame_num: int) -> bool:
        """True if frame metrics not in StatsManager, False otherwise."""
        if self.stats_manager is None:
            return True
        return all([detector.is_processing_required(frame_num) for detector in self._detector_list])

# -*- coding: utf-8 -*-
#
"""
class:`ContentDetector`
比较相邻帧之间的内容差异
设置阈值/分数，如果超过，将触发场景切换
"""
from dataclasses import dataclass
import math

"""
typing 模块中导入了 List、NamedTuple 和 Optional 这三个类。这些类可以用于在代码中明确指定变量、参数、返回值等的类型，以提高代码的可读性和可维护性。
List 是用于表示列表的类型，NamedTuple 是用于创建命名元组的工具，Optional 则表示一个值可以是某种类型或者 None。
"""
from typing import List, NamedTuple, Optional

import numpy
import cv2

from Tools.scene_detector import SceneDetector


def _mean_pixel_distance(left: numpy.ndarray, right: numpy.ndarray) -> float:
    """
    这个函数的目的是计算两个图像 left 和 right 之间的平均像素值距离
    """

    # 确保传递给函数的图像都是二维的。left.shape 和 right.shape 分别表示图像 left 和 right 的形状。如果形状不是二维的，将会引发一个异常。
    assert len(left.shape) == 2 and len(right.shape) == 2

    # 确保传递给函数的两个图像具有相同的形状
    assert left.shape == right.shape

    # 这一行计算了图像中的像素总数，并将其存储在变量 num_pixels 中。left.shape[0] 和 left.shape[1] 分别表示图像的行数和列数
    num_pixels: float = float(left.shape[0] * left.shape[1])

    # 它计算了两个图像中相应像素值的绝对差异，然后求和，并将其除以总像素数以获得平均值。最终的计算结果被返回给调用者
    return (numpy.sum(numpy.abs(left.astype(numpy.int32) - right.astype(numpy.int32))) / num_pixels)


def _estimated_kernel_size(frame_width: int, frame_height: int) -> int:
    """
    根据视频分辨率估算内核大小。

    TODO: 这个方程基于对一些视频的手动估计。
    需要创建一个更全面的测试套件来进行优化。

    Arguments:
        frame_width (int): 视频帧的宽度。
        frame_height (int): 视频帧的高度。

    Returns:
        int: 估计的内核大小。
    """
    # 使用一个基于分辨率的公式来估算内核的大小。
    size: int = 4 + round(math.sqrt(frame_width * frame_height) / 192)

    # 如果大小为偶数，将其增加1，以确保内核大小为奇数。
    if size % 2 == 0:
        size += 1

    return size


class ContentDetector(SceneDetector):
    """
    通过帧间颜色和亮度的变化来检测快速切换。

    由于使用了帧间的差异，与ThresholdDetector不同，
    这种方法只能检测到快速切换。
    若要仍然使用HSV信息检测内容场景之间的缓慢淡入淡出效果，请使用DissolveDetector。
    """

    # TODO: 如果有一个可以通过更多测试用例的新默认值的话，请提供一些好的权重。
    class Components(NamedTuple):
        """构成帧得分的组件及其默认值。"""
        delta_hue: float = 1.0
        """相邻帧的像素色相值之间的差异。"""
        delta_sat: float = 1.0
        """相邻帧的像素饱和度值之间的差异。"""
        delta_lum: float = 1.0
        """相邻帧的像素亮度值之间的差异。"""
        delta_edges: float = 0.0
        """相邻帧的计算边缘之间的差异。

        通常边缘的差异比其他组件大，因此可能需要相应地调整检测阈值。
        """

    DEFAULT_COMPONENT_WEIGHTS = Components()
    """
    默认的组件权重。
    实际的默认值在Components类中指定，以允许在不破坏现有用法的情况下添加新组件。
    """

    LUMA_ONLY_WEIGHTS = Components(
        delta_hue=0.0,
        delta_sat=0.0,
        delta_lum=1.0,
        delta_edges=0.0,
    )
    """如果设置了`luma_only`，则使用的组件权重。"""

    FRAME_SCORE_KEY = 'content_val'
    """
    在统计文件中表示经过指定组件加权后的最终帧得分的键。
    FRAME_SCORE_KEY 是一个字符串，它表示在统计文件中用于标识经过指定组件加权后的最终帧得分的键。这个键的值将用于记录每个帧的得分，以便后续分析和处理。
    例如，如果在一个视频中检测到了多个场景变化，那么对于每个场景变化，都会有一个相应的帧得分与之关联
    """

    METRIC_KEYS = [FRAME_SCORE_KEY, *Components._fields]
    """
    此检测器生成的所有统计文件键。
    METRIC_KEYS 是一个列表，其中包含了所有用于记录在统计文件中的信息的键。
    具体来说，它包括了 FRAME_SCORE_KEY，以及 Components 类中定义的所有组件（例如 delta_hue、delta_sat 等）的键。
    这个列表提供了一个完整的记录指标，可以用于分析视频中的场景变化。
    """

    @dataclass
    class _FrameData:
        """计算给定帧所需的数据。"""
        hue: numpy.ndarray
        """帧的色相映射 [2D 8位]。"""
        sat: numpy.ndarray
        """帧的饱和度映射 [2D 8位]。"""
        lum: numpy.ndarray
        """帧的亮度映射 [2D 8位]。"""
        edges: Optional[numpy.ndarray]
        """帧的边缘映射 [2D 8位，边缘为255，非边缘为0]。受`kernel_size`影响。"""

    def __init__(
        self,
        threshold: float = 27.0,
        min_scene_len: int = 15,
        weights: 'ContentDetector.Components' = DEFAULT_COMPONENT_WEIGHTS,
        luma_only: bool = False,
        kernel_size: Optional[int] = None,
    ):
        """
        Arguments:
            threshold: Threshold the average change in pixel intensity must exceed to trigger a cut.
            min_scene_len: Once a cut is detected, this many frames must pass before a new one can
                be added to the scene list.
            weights: Weight to place on each component when calculating frame score
                (`content_val` in a statsfile, the value `threshold` is compared against).
            luma_only: If True, only considers changes in the luminance channel of the video.
                Equivalent to specifying `weights` as :data:`ContentDetector.LUMA_ONLY`.
                Overrides `weights` if both are set.
            kernel_size: Size of kernel for expanding detected edges. Must be odd integer
                greater than or equal to 3. If None, automatically set using video resolution.
        """
        super().__init__()
        self._threshold: float = threshold
        self._min_scene_len: int = min_scene_len
        self._last_scene_cut: Optional[int] = None
        self._last_frame: Optional[ContentDetector._FrameData] = None
        self._weights: ContentDetector.Components = weights
        if luma_only:
            self._weights = ContentDetector.LUMA_ONLY_WEIGHTS
        self._kernel: Optional[numpy.ndarray] = None
        if kernel_size is not None:
            print(kernel_size)
            if kernel_size < 3 or kernel_size % 2 == 0:
                raise ValueError('kernel_size must be odd integer >= 3')
            self._kernel = numpy.ones((kernel_size, kernel_size), numpy.uint8)
        self._frame_score: Optional[float] = None

    def get_metrics(self):
        return ContentDetector.METRIC_KEYS

    def is_processing_required(self, frame_num):
        return True

    def _calculate_frame_score(self, frame_num: int, frame_img: numpy.ndarray) -> float:
        """Calculate score representing relative amount of motion in `frame_img` compared to
        the last time the function was called (returns 0.0 on the first call)."""
        # TODO: Add option to enable motion estimation before calculating score components.
        # TODO: Investigate methods of performing cheaper alternatives, e.g. shifting or resizing
        # the frame to simulate camera movement, using optical flow, etc...

        # Convert image into HSV colorspace.
        hue, sat, lum = cv2.split(cv2.cvtColor(frame_img, cv2.COLOR_BGR2HSV))

        # Performance: Only calculate edges if we have to.
        calculate_edges: bool = ((self._weights.delta_edges > 0.0)
                                 or self.stats_manager is not None)
        edges = self._detect_edges(lum) if calculate_edges else None

        if self._last_frame is None:
            # Need another frame to compare with for score calculation.
            self._last_frame = ContentDetector._FrameData(hue, sat, lum, edges)
            return 0.0

        score_components = ContentDetector.Components(
            delta_hue=_mean_pixel_distance(hue, self._last_frame.hue),
            delta_sat=_mean_pixel_distance(sat, self._last_frame.sat),
            delta_lum=_mean_pixel_distance(lum, self._last_frame.lum),
            delta_edges=(0.0 if edges is None else _mean_pixel_distance(
                edges, self._last_frame.edges)),
        )

        frame_score: float = (
            sum(component * weight for (component, weight) in zip(score_components, self._weights))
            / sum(abs(weight) for weight in self._weights))

        # Record components and frame score if needed for analysis.
        if self.stats_manager is not None:
            metrics = {self.FRAME_SCORE_KEY: frame_score}
            metrics.update(score_components._asdict())
            self.stats_manager.set_metrics(frame_num, metrics)

        # Store all data required to calculate the next frame's score.
        self._last_frame = ContentDetector._FrameData(hue, sat, lum, edges)
        return frame_score

    def process_frame(self, frame_num: int, frame_img: numpy.ndarray) -> List[int]:
        """ Similar to ThresholdDetector, but using the HSV colour space DIFFERENCE instead
        of single-frame RGB/grayscale intensity (thus cannot detect slow fades with this method).

        Arguments:
            frame_num: Frame number of frame that is being passed.
            frame_img: Decoded frame image (numpy.ndarray) to perform scene
                detection on. Can be None *only* if the self.is_processing_required() method
                (inhereted from the base SceneDetector class) returns True.

        Returns:
            List of frames where scene cuts have been detected. There may be 0
            or more frames in the list, and not necessarily the same as frame_num.
        """
        if frame_img is None:
            # TODO(0.6.3): Make frame_img a required argument in the interface. Log a warning
            # that passing None is deprecated and results will be incorrect if this is the case.
            return []

        # Initialize last scene cut point at the beginning of the frames of interest.
        if self._last_scene_cut is None:
            self._last_scene_cut = frame_num

        self._frame_score = self._calculate_frame_score(frame_num, frame_img)
        if self._frame_score is None:
            return []

        # We consider any frame over the threshold a new scene, but only if
        # the minimum scene length has been reached (otherwise it is ignored).
        min_length_met = (frame_num - self._last_scene_cut) >= self._min_scene_len
        if self._frame_score >= self._threshold and min_length_met:
            self._last_scene_cut = frame_num
            return [frame_num]

        return []

    # TODO(#250): Based on the parameters passed to the ContentDetector constructor,
    # ensure that the last scene meets the minimum length requirement, otherwise it
    # should be merged with the previous scene. This can be done by caching the cuts
    # for the amount of time the minimum length is set to, returning any outstanding
    # final cuts in post_process.

    #def post_process(self, frame_num):
    #    """
    #    return []

    def _detect_edges(self, lum: numpy.ndarray) -> numpy.ndarray:
        """Detect edges using the luma channel of a frame.

        Arguments:
            lum: 2D 8-bit image representing the luma channel of a frame.

        Returns:
            2D 8-bit image of the same size as the input, where pixels with values of 255
            represent edges, and all other pixels are 0.
        """
        # Initialize kernel.
        if self._kernel is None:
            kernel_size = _estimated_kernel_size(lum.shape[1], lum.shape[0])
            self._kernel = numpy.ones((kernel_size, kernel_size), numpy.uint8)

        # Estimate levels for thresholding.
        # TODO(0.6.3): Add config file entries for sigma, aperture/kernel size, etc.
        sigma: float = 1.0 / 3.0
        median = numpy.median(lum)
        low = int(max(0, (1.0 - sigma) * median))
        high = int(min(255, (1.0 + sigma) * median))

        # Calculate edges using Canny algorithm, and reduce noise by dilating the edges.
        # This increases edge overlap leading to improved robustness against noise and slow
        # camera movement. Note that very large kernel sizes can negatively affect accuracy.
        edges = cv2.Canny(lum, low, high)
        return cv2.dilate(edges, self._kernel)

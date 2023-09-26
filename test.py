import sys
import os
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from PyQt5.uic import loadUi


class Picture(QMainWindow):
    def __init__(self, parent=None, url=None):
        super().__init__(parent)
        self.url = url
        self.ui()

    def ui(self):
        loadUi('./show_pic.ui', self)

        self.setFixedSize(850, 600)

        total = len(self.url)

        self.qw = QWidget()
        if total % 5 == 0:
            rows = int(total / 5)
        else:
            rows = int(total / 5) + 1
        self.qw.setMinimumSize(850, 230 * rows)
        for i in range(total):

            photo = QPixmap(url[i])
            # print('photo:',photo)
            # photo.loadFromData(req.content)
            width = photo.width()
            height = photo.height()
            print('width:', width, '      ', 'height:', height)

            if width == 0 or height == 0:
                continue
            tmp_image = photo.toImage()  # 将QPixmap对象转换为QImage对象
            size = QSize(width, height)
            # photo.convertFromImage(tmp_image.scaled(size, Qt.IgnoreAspectRatio))
            photo = photo.fromImage(tmp_image.scaled(size, Qt.IgnoreAspectRatio))
            tmp = QWidget(self.qw)

            vl = QVBoxLayout()

            # 为每个图片设置QLabel容器
            label = QLabel()
            label.setFixedSize(150, 200)
            label.setStyleSheet("border:1px solid gray")
            label.setPixmap(photo)
            label.setScaledContents(True)  # 图像自适应窗口大小

            vl.addWidget(label)

            tmp.setLayout(vl)
            tmp.move(160 * (i % 5), 230 * int(i / 5))

        self.scrollArea.setWidget(self.qw)  # 和ui文件中名字相同


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # 这是我的文件夹中图片的路径
    folder_path = 'C:\\Users\\yienk\\PycharmProjects\\pythonProject2\\images'


    # 初始化一个空列表来存储文件路径
    url = []

    # 遍历文件夹
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            # 获取文件的完整路径
            file_path = os.path.join(root, file_name)

            # 将路径添加到url列表中
            url.append(file_path)

    pic = Picture(url=url[:])
    pic.show()
    sys.exit(app.exec_())

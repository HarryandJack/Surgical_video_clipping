import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QPainter
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog
from mainWindow import *
from childWindow import *


# mainWindow
class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyMainWindow, self).__init__()
        self.setupUi(self)

        self.setGeometry(0, 0, 1024, 600)
        self.setWindowTitle('main window')

    def paintEvent(self, event):
        painter = QPainter(self)
        pixmap = QPixmap("./image/bg.jpg")
        painter.drawPixmap(self.rect(), pixmap)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()


class ChildWindow(QDialog, Ui_Dialog):
    def __init__(self):
        super(ChildWindow, self).__init__()
        self.setupUi(self)

        self.setWindowTitle('child window')

        self.pushButton.clicked.connect(self.btnClick)  # 按钮事件绑定

    def btnClick(self):  # 子窗体自定义事件
        self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = MyMainWindow()

    child = ChildWindow()

    btn = main.pushButton  # 主窗体按钮事件绑定
    btn.clicked.connect(child.show)

    main.show()
    sys.exit(app.exec_())


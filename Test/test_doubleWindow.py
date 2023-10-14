from PyQt5.QtWidgets import *
import sys


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 设置窗口标题和大小
        self.setWindowTitle('TestWindow')
        self.resize(400, 300)

        self.collec_btn = QPushButton('打开新窗口', self)

        layout = QVBoxLayout()
        layout.addWidget(self.collec_btn)
        self.setLayout(layout)

        self.show()


class NewWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('新窗口')
        self.resize(280, 230)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # 创建窗口
    window = MainWindow()
    newWin = NewWindow()

    # 显示窗口
    window.show()
    window.collec_btn.clicked.connect(newWin.show)
    # 运行应用，并监听事件
    sys.exit(app.exec_())
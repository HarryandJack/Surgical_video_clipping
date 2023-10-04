from show_pic import Ui_MainWindow
import sys
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QApplication
class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.Plus.clicked.connect(self.test)
    def test(self):
        num_1 = int(self.num_1.text())
        num_2 = int(self.num_2.text())
        num = num_1 + num_2
        self.sum.setText(str(num))  # 将结果显示在sum的文本框中

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MyWindow = MyWindow()
    MyWindow.show()
    sys.exit(app.exec_())
import os
import sys
from datetime import datetime

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, \
    QMessageBox, QLineEdit, QScrollArea


class TrimVideo(QWidget):

    def __init__(self):
        super(TrimVideo, self).__init__()  # 继承父类（QWidget）的__init__()方法

        self.initUI()  # 调用initUI()方法，设置UI
        self.clip_num = 0  # 切片数量
        self.timelist = []  # 记录切片对应的起始、终止时间
        self.file_type, self.file_name, self.videoname, self.file_path = None, None, None, None

    def initUI(self):
        # 设置窗口标题为"视频修剪"
        self.setWindowTitle('视频修剪')

        # 设置视频名称标签和输入框
        self.videoname_label = QLabel('视频', self)
        self.videoname_input = QLineEdit(self)
        # 设置浏览按钮，并连接select_video函数
        self.videoname_btn = QPushButton('浏览', self)
        self.videoname_btn.clicked.connect(self.select_video)

        # 设置剪辑片段数量标签和输入框
        self.clip_label = QLabel('切片数量', self)
        self.clip_input = QLineEdit(self)
        # 设置输入按钮，并连接input_clip函数
        self.clip_btn = QPushButton('输入', self)
        self.clip_btn.clicked.connect(self.input_clip)

        # HBox横向布局，设置视频名称相关组件（一个盒子模型，从左到右依次是此项标题、输入框、按钮）
        hbox_videoname = QHBoxLayout()
        hbox_videoname.addWidget(self.videoname_label)
        hbox_videoname.addWidget(self.videoname_input)
        hbox_videoname.addWidget(self.videoname_btn)

        # 横向布局，设置剪辑片段相关组件
        hbox_clip = QHBoxLayout()
        hbox_clip.addWidget(self.clip_label)
        hbox_clip.addWidget(self.clip_input)
        hbox_clip.addWidget(self.clip_btn)

        self.scroll_area = QScrollArea()
        self.scroll_area_widget = QWidget()
        self.scroll_area_widget.setLayout(QVBoxLayout())
        self.scroll_area.setWidget(self.scroll_area_widget)
        self.scroll_area.setWidgetResizable(True) # 内容自适应大小
        self.scroll_area.setFixedHeight(500) # 滚动区高度

        # VBox竖向布局，添加填充空间和横向布局
        self.vbox = QVBoxLayout()
        self.vbox.addStretch(1)  # 添加填充空间
        self.vbox.addWidget(self.scroll_area) # 添加滚动区，放置后续动态添加的起止时间框和go按钮等
        self.vbox.addLayout(hbox_videoname)  # 添加横向布局，视频名称相关组件
        self.vbox.addLayout(hbox_clip)  # 添加横向布局，剪辑片段相关组件

        # 设置布局
        self.setLayout(self.vbox)

        # 定时器
        self.timer = QTimer(self)
        self.timer.setInterval(3 * 60 * 1000)  # 3分钟，单位：ms
        self.timer.timeout.connect(self.close) # 超时关闭运行

        # 设置窗口位置及大小，显示窗口
        self.setGeometry(300, 300, 400, 600)
        self.show()
        # 启动定时器
        self.timer.start()


    def select_video(self):
        default_fault = r'.'  # 默认打开文件时显示的文件夹路径
        videoget = QFileDialog.getOpenFileName(self, '选择视频', default_fault)  # 打开对话框，标题为"选择视频"
        if videoget[0] != '':  # 如果选择的文件不为空
            self.videoname_input.setText(videoget[0])  # 将文件路径显示到videoname_input
            self.videoname = videoget[0]  # 将文件路径赋值给self.videoname


    def input_clip(self):
        clip_num_str = self.clip_input.text()
        if clip_num_str.isnumeric():
            self.clip_num = int(self.clip_input.text())  # 获取切片数
            if self.clip_num > 0:
                # 重新点击enter时，清除滚动区中已存在的起止框和go按钮
                while self.scroll_area_widget.layout().count():
                    item = self.scroll_area_widget.layout().takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()
                        self.timelist = []
                    elif item.layout():
                        while item.layout().count():
                            sub_item = item.layout().takeAt(0)
                            if sub_item.widget():
                                sub_item.widget().deleteLater()
                            self.timelist = []
                    else:
                        break

                for i in range(self.clip_num):  # 遍历所有切片，每个切片都有起止时间输入框
                    start_label = QLabel(f'start {i + 1}')  # 设置start框的label（此项标题）
                    start_input = QLineEdit("00:00:00")  # 设置start框的输入框，默认输入值为00:00:00
                    end_label = QLabel(f'end {i + 1}')
                    end_input = QLineEdit("00:00:00")
                    self.timelist.append([start_input, end_input])  # 把输入框加到self.timelist

                    hbox_temp = QHBoxLayout()  # 生成临时组件
                    hbox_temp.addWidget(start_label) # 把标题和输入框都加进临时组件
                    hbox_temp.addWidget(start_input)
                    hbox_temp.addWidget(end_label)
                    hbox_temp.addWidget(end_input)
                    self.scroll_area_widget.layout().addLayout(hbox_temp) # 把临时组件加入界面

                self.go_btn = QPushButton('开始', self) # 生成按钮
                self.go_btn.clicked.connect(self.start_trimming) # 定义开始按钮的点击操作，调用start_trimming函数
                self.scroll_area_widget.layout().addWidget(self.go_btn) # 把开始按钮加入界面
            else:
                QMessageBox.warning(self, '错误', '切片数量必须大于0！')
        else:
            QMessageBox.warning(self, '错误', '切片数量需为数字！')


    def start_trimming(self):
        # 检查输入合法性
        for i in range(self.clip_num):
            start_time = self.timelist[i][0].text()  # 获取开始时间
            end_time = self.timelist[i][1].text()  # 获取结束时间
            # 检查时间
            if datetime.strptime(start_time, "%H:%M:%S") >= datetime.strptime(end_time, "%H:%M:%S"):
                QMessageBox.warning(self, "错误", "请保证格式为%H:%M:%S，且开始时间小于结束时间")
                return

        cmd_txt = ".\store\cmd.txt"  # 命令行执行信息存储的位置
        concat_txt = ".\concat.txt"  # 待合并的切片信息存储的位置

        file, self.file_type = self.videoname.split('.')
        self.file_name = self.videoname.split('/')[-1].split('.')[0]
        self.file_path = self.videoname.split('/')[:-1]
        save_path = '/'.join(self.file_path)
        if save_path == '' or save_path == None:
            QMessageBox.critical(self, '错误', '请选择正确的保存路径！')
            return

        for i in range(self.clip_num):  # 遍历每个切片
            temp_video_name = save_path + '/' + self.file_name + '_' + str(i) + '.' + self.file_type  # 切片视频的名字
            start_time = self.timelist[i][0].text()  # 取出开始时间
            end_time = self.timelist[i][1].text()  # 取出结束时间
            cmd = f'ffmpeg -y -i "{self.videoname}" -ss {start_time} -to {end_time} -c copy -hide_banner "{temp_video_name}" > {cmd_txt} 2>&1'
            os.system(cmd)  # 执行切片命令
        output_video_name = save_path + '/' + self.file_name + '_trimmed' + '.' + self.file_type  # 新视频的完整名字
        with open(concat_txt, 'w', encoding="utf-8") as f:  # 存储分段信息到concat.txt
            for i in range(self.clip_num):
                f.write(f"file '{save_path}/{self.file_name}_{i}.{self.file_type}'\n")
        cmd = f'ffmpeg -y -f concat -safe 0 -i {concat_txt} -c copy "{output_video_name}" > {cmd_txt} 2>&1'
        os.system(cmd)  # 执行合并命令
        # 删除切片视频
        for i in range(self.clip_num):
            file_to_delete = f"{save_path}/{self.file_name}_{i}.{self.file_type}"
            os.remove(file_to_delete)

        QMessageBox.information(self, '提示', '修剪完成！')


    def closeEvent(self, event):
        super().closeEvent(event)

    def mousePressEvent(self, event):
        self.timer.start()

    def keyPressEvent(self, event):
        self.timer.start()

    def wheelEvent(self, event):
        self.timer.start()


if __name__ == '__main__':
    # 创建QApplication类的实例
    app = QApplication(sys.argv)
    # 创建TrimVideo类的实例
    ex = TrimVideo()
    # 退出程序，直到主事件循环完成
    sys.exit(app.exec_())
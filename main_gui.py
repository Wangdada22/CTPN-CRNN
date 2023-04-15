import os
from ocr import ocr
import time
import shutil
import numpy as np
from PIL import Image
from glob import glob
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from MyUI.mainwindow import Ui_MainWindow
import sys
from PyQt5.QtWidgets import QFileDialog, QApplication


class myWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        # QtCreater 设计界面.ui->.py pyuic5 -o mainwindow.py mainwindow.ui
        self.setupUi(self)
        #self.timer_camera = QtCore.QTimer()
        #self.cap = cv2.VideoCapture()
        #self.CAM_NUM = 0
        self.dataset = ""
        self.eval_loaders = ""
        self.num = 1
        self.text = 't'
        self.slot_init()

    def slot_init(self):
        # self.button_open_camera.clicked.connect(self.get_data)    #打开摄像头按键
        # self.timer_camera.timeout.connect(self.read_videostream)                          #定时器结束，则调用read_videostream()
        self.button_close.clicked.connect(
            QtCore.QCoreApplication.instance().quit)
        self.button_open_file.clicked.connect(
            self.openImage)  # 选择图像或视频进行识别

    def single_pic_proc(self, image_file):
        image = np.array(Image.open(image_file).convert('RGB'))
        result, image_framed = ocr(image)
        return result, image_framed

    def saveImage(self):  # 保存图片到本地
        screen = QApplication.primaryScreen()
        pix = screen.grabWindow(self.label_image.winId())
        fd, type = QFileDialog.getSaveFileName(
            self.centralwidget, "保存图片", "", "*.jpg;;*.png;;All Files(*)")
        pix.save(fd)

    def openDirectory(self):  # 打开文件夹（目录）
        fd = QFileDialog.getExistingDirectory(self.centralwidget, "选择文件夹", "")
        # self.label_directoryPath.setText(fd)
        self.detect_img(fd)

    def openImage(self):  # 选择本地图片上传
        # 弹出一个文件选择框，第一个返回值imgName记录选中的文件路径+文件名，第二个返回值imgType记录文件的类型
        imgName, imgType = QFileDialog.getOpenFileName(
            self.centralwidget, "打开图片", "", "*.png;;*.jpg;;All Files(*)")
        self.detect_img(imgName)
        # jpg = QtGui.QPixmap(imgName).scaled(self.label_image.width(
        # ), self.label_image.height())  # 通过文件路径获取图片文件，并设置图片长宽为label控件的长款
        # self.label_image.setPixmap(jpg)  # 在label控件上显示选择的图片

    def detect_img(self, img_path):
        self.text_show.clear()
        result, image_framed = self.single_pic_proc(img_path)
        print(type(result))
        for key in result:
            #result_text = result_text+str(result[key][1])
            self.text_show.append(str(result[key][1]))
            # print(result[key][1])
        # 显示结果图片
        png = QtGui.QPixmap(img_path).scaled(
            self.label_show_camera.width(), self.label_show_camera.height())
        self.label_show_camera.setPixmap(png)

    def select_file(self):
        # test_images\t1.png
        self.text_show.clear()
        result_text = ""
        img_path = 'test_images/t{}.png'.format(self.num)
        result, image_framed = self.single_pic_proc(img_path)
        print(type(result))
        for key in result:
            #result_text = result_text+str(result[key][1])
            self.text_show.append(str(result[key][1]))
            # print(result[key][1])
        # 显示结果图片
        png = QtGui.QPixmap(img_path).scaled(
            self.label_show_camera.width(), self.label_show_camera.height())
        self.label_show_camera.setPixmap(png)
        # self.text_show.append(str(result_text))
        self.num = self.num+1


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = myWindow()
    win.show()
    sys.exit(app.exec())




import sys
import cv2
from PyQt5 import QtCore, QtGui,QtWidgets
from gui.mainwindow import Ui_MainWindow



import numpy as np
from tensorflow.keras.models import load_model
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.inference import load_image
from utils.preprocessor import preprocess_input
from statistics import mode
#from emotion_gender import *
import pygame
import os
import random
import time
# starting lists for calculating modes
emotion_window = []

# starting video streaming
#cv2.namedWindow('window_frame')
# parameters for loading data and images
#image_path = sys.argv[1]
#imge_path = ""
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
gender_model_path = '../trained_models/gender_models/simple_CNN.81-0.96.hdf5'
emotion_labels = get_labels('fer2013')
gender_labels = get_labels('imdb')
font = cv2.FONT_HERSHEY_SIMPLEX

# hyper-parameters for bounding boxes shape
frame_window = 10
gender_offsets = (30, 60)
gender_offsets = (10, 10)
emotion_offsets = (20, 40)
emotion_offsets = (0, 0)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
gender_classifier = load_model(gender_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]
gender_target_size = gender_classifier.input_shape[1:3]



class myWindow(QtWidgets.QMainWindow,Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setupUi(self)
        self.timer_camera = QtCore.QTimer()
        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 0
        self.slot_init()

    def slot_init(self):
        self.button_open_camera.clicked.connect(self.button_open_camera_clicked)    #打开摄像头按键
        self.timer_camera.timeout.connect(self.read_videostream)                          #定时器结束，则调用read_videostream()
        self.button_close.clicked.connect(QtCore.QCoreApplication.instance().quit)
        self.button_open_file.clicked.connect(self.select_file)         #选择图像或视频进行识别

    #def detect_face(self,img_d):

    def button_open_camera_clicked(self):
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self,'warning','请检查相机是否正常',buttons = QtWidgets.QMessageBox.OK)
            else:
                self.timer_camera.start(30)#定时器开始计时，每30ms从摄像头读取一帧图像
                self.button_open_camera.setText('Close Camera!')
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.label_show_camera.clear()
            self.button_open_camera.setText('Open Camera!')

    def select_file(self):
       # m = QtWidgets.QFileDialog.get(None,"选取文件夹","D:/")#起始路径
        imgName,imgType = QtWidgets.QFileDialog.getOpenFileName(self,"打开图片","","*.jpg;;*.png;;All Files(*)")
        show_img = gender_recognition(imgName)
        #show_img = cv2.resize()
        
        self.show_image(show_img)
        #self.file_address.setText(m)
        #print(imgName,imgType)

        #Img = QtGui.QPixmap(imgName).scaled(self.label_show_camera.width(),self.label_show_camera.height(),aspectRatioMode=1,transformMode=1)#等比例缩放
        #self.label_show_camera.setScaledContents(True)

        #print(type(Img))
        #Img = QtGui.QPixmap(imgName)
        #self.label_show_camera.setPixmap(Img)

    #实时视频流读取
    def read_videostream(self):
        flag,self.frame = self.cap.read()#read from video stream
        #show_img = self.detect_face(self.frame)#
        
        show_img = self.emotion_recognition()
        
        
        #img_d = cv2.resize(self.frame,(640,480))            #把读取到的帧的大小重新设置为640x480
        #img_d = cv2.cvtColor(img_d,cv2.COLOR_BGR2RGB)       #视频色彩转回RGB
        #grey = cv2.cvtColor(img_d,cv2.COLOR_BGR2GRAY)
        #show_img  = img_d
        #show vide on the gui
        self.show_image(show_img)

    #表情识别
    def emotion_recognition(self):
        while True:
            #bgr_image = video_capture.read()[1]
            bgr_image = self.frame
            gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            faces = detect_faces(face_detection, gray_image)

            for face_coordinates in faces:

                x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
                gray_face = gray_image[y1:y2, x1:x2]
                try:
                    gray_face = cv2.resize(gray_face, (emotion_target_size))
                except:
                    continue

                gray_face = preprocess_input(gray_face, True)
                gray_face = np.expand_dims(gray_face, 0)
                gray_face = np.expand_dims(gray_face, -1)
                emotion_prediction = emotion_classifier.predict(gray_face)
                emotion_probability = np.max(emotion_prediction)
                emotion_label_arg = np.argmax(emotion_prediction)
                emotion_text = emotion_labels[emotion_label_arg]
                emotion_window.append(emotion_text)
            
                if len(emotion_window) > frame_window:
                    emotion_window.pop(0)
                try:
                    emotion_mode = mode(emotion_window)
                except:
                    continue
                pygame.mixer.init()
            
        
                if emotion_text == 'angry':
                    color = emotion_probability * np.asarray((255, 0, 0))
                   # self.text_show.append(emotion_text+playMusic)
                    
                elif emotion_text == 'sad' and emotion_probability > 0.8:
                    color = emotion_probability * np.asarray((0, 0, 255))
                    folder = r'D:\Desktop\face_classification\src\music\sad'
                    musics = [folder+'\\'+music for music in os.listdir(folder) if music.endswith('.mp3')]
                    pygame.mixer.music.pause()
                    total =len(musics)
                    if not pygame.mixer.music.get_busy():
                        playMusic = random.choice(musics)
                        pygame.mixer.music.load(playMusic)
                        pygame.mixer.music.play(1)
                        print ("playing...",playMusic)
                        self.text_show.append(emotion_text+playMusic)
                        
                    else:
                        time.sleep(1)
                elif emotion_text == 'happy' and emotion_probability > 0.9:
                    pygame.mixer.music.pause()
                    color = emotion_probability * np.asarray((255, 255, 0))
                    folder = r'D:\Desktop\face_classification\src\music\happy'
                    musics = [folder+'\\'+music for music in os.listdir(folder) if music.endswith('.mp3')]
                    if not pygame.mixer.music.get_busy():
                        playMusic = random.choice(musics)
                        pygame.mixer.music.load(playMusic)
                        pygame.mixer.music.play(1)
                        print ("playing...",playMusic)
                        self.text_show.append("当前的表情是"+emotion_text+"    推荐相应歌曲"+playMusic)
                        
                    else:
                        time.sleep(1)
                elif emotion_text == 'surprise' and emotion_probability > 0.8:
                    color = emotion_probability * np.asarray((0, 255, 255))
                    pygame.mixer.music.pause()

                else:
                    color = emotion_probability * np.asarray((0, 255, 0))
                    pygame.mixer.music.unpause()

                color = color.astype(int)
                color = color.tolist()

                draw_bounding_box(face_coordinates, rgb_image, color)
                draw_text(face_coordinates, rgb_image, emotion_mode,
                        color, 0, -30, 1, 1)
                draw_text(face_coordinates,rgb_image,str(emotion_probability),
                color,0, -50 ,1 ,1)

            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
           
            return rgb_image
            
     #界面显示
    def show_image(self,show_img):
        show_roi = QtGui.QImage(show_img.data,show_img.shape[1],show_img.shape[0],QtGui.QImage.Format_RGB888)
        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(show_roi))

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win  = myWindow()
    win.show()
    sys.exit(app.exec())






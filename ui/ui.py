import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsPixmapItem, QStyle
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from PyQt5 import QtCore

from ui.ui_class import *
# from ui_class import *

import cv2


# 敌方1、2、3、4、5、前哨站、哨兵; 我方~
def Change_blood(ui, blood, ENEMY_COLOR):
    # 敌方颜色：1红、2蓝
    if ENEMY_COLOR == 1:

        ui.enemy1.setProperty("value", blood[0])
        ui.enemy2.setProperty("value", blood[1])
        ui.enemy3.setProperty("value", blood[2])
        ui.enemy4.setProperty("value", blood[3])
        ui.enemy5.setProperty("value", blood[4])
        ui.enemywatch.setProperty("value", blood[5])
        ui.enemypost.setProperty("value", blood[6])
        ui.enemybase.setProperty("value", blood[7])

        ui.our1.setProperty("value", blood[8])
        ui.our2.setProperty("value", blood[9])
        ui.our3.setProperty("value", blood[10])
        ui.our4.setProperty("value", blood[11])
        ui.our5.setProperty("value", blood[12])
        ui.ourwatch.setProperty("value", blood[13])
        ui.ourpost.setProperty("value", blood[14])
        ui.ourbase.setProperty("value", blood[15])

    elif ENEMY_COLOR == 2:

        ui.enemy1.setProperty("value", blood[8])
        ui.enemy2.setProperty("value", blood[9])
        ui.enemy3.setProperty("value", blood[10])
        ui.enemy4.setProperty("value", blood[11])
        ui.enemy5.setProperty("value", blood[12])
        ui.enemywatch.setProperty("value", blood[13])
        ui.enemypost.setProperty("value", blood[14])
        ui.enemybase.setProperty("value", blood[15])

        ui.our1.setProperty("value", blood[0])
        ui.our2.setProperty("value", blood[1])
        ui.our3.setProperty("value", blood[2])
        ui.our4.setProperty("value", blood[3])
        ui.our5.setProperty("value", blood[4])
        ui.ourwatch.setProperty("value", blood[5])
        ui.ourpost.setProperty("value", blood[6])
        ui.ourbase.setProperty("value", blood[7])


# 初始化/更新敌方英雄/步兵的最大血量
def Init_blood(ui, blood, ENEMY_COLOR):
    if ENEMY_COLOR == 1:
        ui.enemy1.setProperty("maximum", blood[0])
        ui.enemy3.setProperty("maximum", blood[2])
        ui.enemy4.setProperty("maximum", blood[3])
        ui.enemy5.setProperty("maximum", blood[4])

        ui.our1.setProperty("maximum", blood[8])
        ui.our3.setProperty("maximum", blood[10])
        ui.our4.setProperty("maximum", blood[11])
        ui.our5.setProperty("maximum", blood[12])


    elif ENEMY_COLOR == 2:

        ui.enemy1.setProperty("maximum", blood[8])
        ui.enemy3.setProperty("maximum", blood[10])
        ui.enemy4.setProperty("maximum", blood[11])
        ui.enemy5.setProperty("maximum", blood[12])

        ui.our1.setProperty("maximum", blood[0])
        ui.our3.setProperty("maximum", blood[2])
        ui.our4.setProperty("maximum", blood[3])
        ui.our5.setProperty("maximum", blood[4])

    Change_blood(ui, blood, ENEMY_COLOR)


# 敌方颜色：1红、2蓝
def Color_blood(ui, ENEMY_COLOR):
    red_file = open("./ui/StyleSheet_red_progressbar.txt", 'r')
    blue_file = open("./ui/StyleSheet_blue_progressbar.txt", 'r')

    red_sheet = red_file.read(-1)
    blue_sheet = blue_file.read(-1)

    if ENEMY_COLOR == 1:
        ui.enemy1.setStyleSheet(red_sheet)
        ui.enemy2.setStyleSheet(red_sheet)
        ui.enemy3.setStyleSheet(red_sheet)
        ui.enemy4.setStyleSheet(red_sheet)
        ui.enemy5.setStyleSheet(red_sheet)
        ui.enemywatch.setStyleSheet(red_sheet)
        ui.enemypost.setStyleSheet(red_sheet)
        ui.enemybase.setStyleSheet(red_sheet)

        ui.our1.setStyleSheet(blue_sheet)
        ui.our2.setStyleSheet(blue_sheet)
        ui.our3.setStyleSheet(blue_sheet)
        ui.our4.setStyleSheet(blue_sheet)
        ui.our5.setStyleSheet(blue_sheet)
        ui.ourwatch.setStyleSheet(blue_sheet)
        ui.ourpost.setStyleSheet(blue_sheet)
        ui.ourbase.setStyleSheet(blue_sheet)

    elif ENEMY_COLOR == 2:
        ui.enemy1.setStyleSheet(blue_sheet)
        ui.enemy2.setStyleSheet(blue_sheet)
        ui.enemy3.setStyleSheet(blue_sheet)
        ui.enemy4.setStyleSheet(blue_sheet)
        ui.enemy5.setStyleSheet(blue_sheet)
        ui.enemywatch.setStyleSheet(blue_sheet)
        ui.enemypost.setStyleSheet(blue_sheet)
        ui.enemybase.setStyleSheet(blue_sheet)

        ui.our1.setStyleSheet(red_sheet)
        ui.our2.setStyleSheet(red_sheet)
        ui.our3.setStyleSheet(red_sheet)
        ui.our4.setStyleSheet(red_sheet)
        ui.our5.setStyleSheet(red_sheet)
        ui.ourwatch.setStyleSheet(red_sheet)
        ui.ourpost.setStyleSheet(red_sheet)
        ui.ourbase.setStyleSheet(red_sheet)

    red_file.close()
    blue_file.close()


def Color_time(ui, alarm):
    red_file = open("./ui/StyleSheet_red_time.txt", 'r')
    black_file = open("./ui/StyleSheet_black_time.txt", 'r')

    red_sheet = red_file.read(-1)
    black_sheet = black_file.read(-1)

    if alarm == True:
        ui.time.setStyleSheet(red_sheet)
    else:
        ui.time.setStyleSheet(black_sheet)


def Change_time(ui, min, sec):
    ui.time.setDateTime(QtCore.QDateTime(QtCore.QDate(2000, 1, 1), QtCore.QTime(0, min, sec)))


def Change_text(ui, text):
    _translate = QtCore.QCoreApplication.translate

    html_prefix = """<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n
<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n
p, li { white-space: pre-wrap; }\n
</style></head><body style=\" font-family:\'Adobe Devanagari\'; font-size:9pt; font-weight:400; font-style:normal;\">\n
<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:28pt;\">"""

    html_suffix = "</span></p></body></html>"

    html = html_prefix + text + html_suffix

    ui.Text.setHtml(_translate("MainWindow", html))


def Clear_text(ui):
    Change_text(ui, '')


if __name__ == '__main__':
    app = QApplication(sys.argv)

    exit_flag = False

    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)

    blood = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

    Color_blood(ui, 1)

    Change_blood(ui, blood)

    Color_time(ui, True)

    Change_text(ui, "准备动手准备动手！")


    # Change_text(ui,"")

    def onClick_Button():
        # 退出应用程序
        global exit_flag
        exit_flag = True
        print('exit')
        app.exit()
        MainWindow.close()


    # ui.pushButton.clicked.connect(onClick_Button)

    scene = QGraphicsScene()

    # scene.updat
    ui.image.setScene(scene)
    ui.image.show()
    MainWindow.show()

    # 使用opencv 打开图片
    # cv2转为QImage
    # QImage转为QPixmap
    # 把QPixmap加入到QGraphicsScene
    # 把QGraphicsScene加入到graphicsView
    # graphicsView show

    # cam = cv2.VideoCapture(1)

    # cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 设置图像宽度
    # cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)  # 设置图像高度
    # cam.set(cv2.CAP_PROP_FPS , 60) 
    # cam.set(cv2.CAP_PROP_EXPOSURE, 10000)
    # cam.set(cv2.CAP_PROP_GAIN, 12)

    # w,h = 1325,1060

    # while not exit_flag:

    #     ret,img = cam.read()

    #     # 若resize过大会自动退出
    #     img = cv2.resize(img,(1325,1060))
    #     img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #     y, x = img.shape[:-1]

    #     frame = QImage(img, x, y,x*3, QImage.Format_RGB888)

    #     scene.clear()  #先清空上次的残留

    #     pix = QPixmap.fromImage(frame)

    #     scene.addPixmap(pix)

    #     cv2.waitKey(1)

    # print('out')
    sys.exit(app.exec_())

'''
class Thread(QThread):

    _signal = pyqtSignal(object)

    def __init__(self,cam):
        super().__init__()

        self.cam = cam
        self.app = QApplication(sys.argv)

        self.exit_flag = False

        self.MainWindow = QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.MainWindow)

        def onClick_Button():

            #退出应用程序
            self.exit_flag = True
            print('exit')
            
            exit(0)



        self.ui.pushButton.clicked.connect(onClick_Button)

        self.scene = QGraphicsScene()

        # scene.updat
        self.ui.image.setScene(self.scene)
        self.ui.image.show()
        self.MainWindow.show()

    def run(self,):
        
        while not self.exit_flag:
            print(1)

            ret,img = self.cam.read()
            
            # 若resize过大会自动退出
            img = cv2.resize(img,(1325,1060))
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            y, x = img.shape[:-1]

            frame = QImage(img, x, y,x*3, QImage.Format_RGB888)

            self.scene.clear()  #先清空上次的残留
            
            pix = QPixmap.fromImage(frame)

            self.scene.addPixmap(pix)

            cv2.waitKey(1)
        self._signal.emit('exit!!')

    def stop(self):
        self.exit_flag = True

# cam = cv2.VideoCapture(1)

# cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 设置图像宽度
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)  # 设置图像高度
# cam.set(cv2.CAP_PROP_FPS , 60) 
# cam.set(cv2.CAP_PROP_EXPOSURE, 10000)
# cam.set(cv2.CAP_PROP_GAIN, 12)

# thread = Thread(cam)
# thread.start()
# thread.exec()

'''

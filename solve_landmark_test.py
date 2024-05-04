# from queue import Queue
from anchor import Anchor
from anchor import set_by_hand
from macro import *

print('initializing...')
import cv2
# from threading import Thread
from threading import Lock
from multiprocessing import Process
import os
import sys
import csv
import time
import numpy as np

from ultralytics import YOLO
import torch
from ruamel.yaml import YAML
import my_serial as messager
from stereo_camera import binocular_camera as bc
from stereo_camera.coex_matcher import CoExMatcher
# from apriltag_locater.apriltag_detector import ApriltagDetector
# import apriltag_locater.apriltag_detector as ad
from hik_camera import camera as mc
from target import Targets, Car, Armor
import coordinate_conversion as cc
# import ui.ui as visualize
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import gui
import alarm

from radar_utils.chessboard_corner import find_chessboard_corners

print('[main] modules imported')

CarsTotal = 5
RedCarsID = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
BlueCarsID = {1: 101, 2: 102, 3: 103, 4: 104, 5: 105}
classes = ["car", "armor1red", "armor2red", "armor3red", "armor4red", "armor5red",
           "armor1blue", "armor2blue", "armor3blue", "armor4blue", "armor5blue", "base", "ignore"]
# import global_var as gv
# 敌方颜色：1红、2蓝
# RED = 0
# BLUE = 1

ENEMY_COLOR = BLUE
portx = 'COM3'
device = torch.device('cuda:0')

main_cfg_path = "./main_config.yaml"
binocular_camera_cfg_path = "bin_cam_config.yaml"
monocular_camera_cfg_path = "mon_cam_config.yaml"
main_cfg = YAML().load(open(main_cfg_path, encoding='Utf-8', mode='r'))
bin_cam_cfg = YAML().load(open(binocular_camera_cfg_path, encoding='Utf-8', mode='r'))
mon_cam_cfg = YAML().load(open(monocular_camera_cfg_path, encoding='Utf-8', mode='r'))

if main_cfg['debug']:
    portx = 'COM7'

arc_roll = bin_cam_cfg['set']['roll'] * np.pi / 180
cos_arc_roll = np.cos(arc_roll)
cam_bias = bin_cam_cfg['set']['bias']

if ENEMY_COLOR == RED:  # self = BLUE
    Enemy_Car_List = [1, 2, 3, 4, 5]
    Own_Car_List = [101, 102, 103, 104, 105]
    Encirclement_List = [101, 103, 104, 105, 107]
    Guard = 107
    Radar = 109
    label, seed, buf = 1, 11200, 14150
    """bin_cam_pos = cc.CameraPose(28988.19 + bin_cam_cfg['set']['bias_x'],
                                6017.49 - bin_cam_cfg['set']['bias_y'],
                                2497 + bin_cam_cfg['set']['bias_y'],
                                0, 0, 0)"""
    radar_base = [28988.19, 6017.49, 2500]  # blue base
    cam_pos = [radar_base[0] + cam_bias[0], radar_base[1] - cam_bias[1], radar_base[2] + cam_bias[2]]
    sign = -1
    ALLY_COLOR = BLUE

elif ENEMY_COLOR == BLUE:  # self = RED
    Enemy_Car_List = [101, 102, 103, 104, 105]
    Own_Car_List = [1, 2, 3, 4, 5]
    Encirclement_List = [1, 3, 4, 5, 7]
    Guard = 7
    Radar = 9
    label, seed, buf = 101, 17942, 1799
    """bin_cam_pos = cc.CameraPose(-987.55 - bin_cam_cfg['set']['bias_x'],
                                9018.02 + bin_cam_cfg['set']['bias_y'],
                                2497 + bin_cam_cfg['set']['bias_y'],
                                0, 0, 0)"""
    radar_base = [-987.55, 9018.02, 2500]  # red base
    cam_pos = [radar_base[0] - cam_bias[0], radar_base[1] + cam_bias[1], radar_base[2] + cam_bias[2]]
    sign = 1
    ALLY_COLOR = RED

# 核心变量：EnemyCars数组
# 临时变量：用于储存预测结果为Car的object，使用完立即clear
targets = Targets(ENEMY_COLOR)
Enemy_Cars = []
count_down = 420
blood = [0 for i in range(16)]
guard_location = (0, 0)

# 全局线程标志位
exit_signal = False
map_sending = False
Alarming = True
time_flag = False
blood_flag = True
blood_init = False

# if cfg['ctrl']['SENDING'] | cfg['ctrl']['RECEIVING']:
ser = messager.serial_init(portx)

# 数据保存
if main_cfg['ctrl']['SAVE_IMG']:
    time_now = time.localtime()
    img_folder = './record' + str(time_now[1]) + '-' + str(time_now[2]) + '-' + str(time_now[3]) + '-' + str(
        time_now[4])  # months + days + hours
    primal_folder = img_folder + '/primal'
    result_folder = img_folder + '/result'
    if not os.path.exists(img_folder):
        os.mkdir(img_folder)
        os.mkdir(primal_folder)
        os.mkdir(result_folder)

if main_cfg['ctrl']['RECORDING']:
    time_now = time.localtime()
    video_folder = './video_record' + str(time_now[1]) + '-' + str(time_now[2]) + '-' + \
                   str(time_now[3]) + '-' + str(time_now[4]) + '-' + str(time_now[5]) + \
                   '-' + str(time_now[6])  # months-days-hours-mins-secs
    raw_video_folder = video_folder + '/raw'
    left_video_folder = raw_video_folder + '/left'
    right_video_folder = raw_video_folder + '/right'
    lf_video_folder = raw_video_folder + '/lf'
    # result_video_folder = video_folder + '/result'
    if not os.path.exists(video_folder):
        os.mkdir(video_folder)
    if not os.path.exists(raw_video_folder):
        os.mkdir(raw_video_folder)
    if not os.path.exists(left_video_folder):
        os.mkdir(left_video_folder)
    if not os.path.exists(right_video_folder):
        os.mkdir(right_video_folder)
    if not os.path.exists(lf_video_folder):
        os.mkdir(lf_video_folder)
    if not os.path.exists(video_folder + '/cfg'):
        os.mkdir(video_folder + '/cfg')

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # ('M', 'P', '4', 'V')
    frame_size = (bin_cam_cfg['param']['Width'], bin_cam_cfg['param']['Height'])
    left_video = cv2.VideoWriter(left_video_folder + "/raw_left.mp4", fourcc, 12, frame_size, True)
    right_video = cv2.VideoWriter(right_video_folder + "/raw_right.mp4", fourcc, 12, frame_size, True)
    depth_video = cv2.VideoWriter(video_folder + "/dep_view_left.mp4", fourcc, 12, frame_size, True)
    disp_video = cv2.VideoWriter(video_folder + "/disp_left.mp4", fourcc, 12, frame_size, True)
    lf_video = cv2.VideoWriter(lf_video_folder + "/lf.mp4", fourcc, 12, frame_size, True)


Loop = True


# 主函数
def main():
    global targets

    print("\nLoading binocular camera")
    camera_left, camera_right, ret_p, ret_q = bc.get_camera(bin_cam_cfg)
    print("Done")

    print("\nLoading matching model")
    coex_matcher = CoExMatcher(bin_cam_cfg)
    print("Done\n")

    if main_cfg['ctrl']['HIGHWAY']:
        print("\nLoading long-focal camera")
        camera_lf, ret_o = mc.get_camera(mon_cam_cfg)
        print("Done")

    left_cam_cfg = dict()
    left_cam_cfg['intrinsic'] = bin_cam_cfg['calib']['intrinsic1']
    left_cam_cfg['distortion'] = bin_cam_cfg['calib']['distortion1']
    camera_pose_solver = cc.CameraPoseSolver(ALLY_COLOR, left_cam_cfg)
    if main_cfg['ctrl']['ANCHOR']:
        anchor = Anchor()
        while True:  # this while is for case where no img got
            image_left = bc.get_frame(camera_left, 'left_camera', ret_p)
            image_right = bc.get_frame(camera_right, 'right_camera', ret_q)

            if image_left is None:
                continue
            set_by_hand(image_left, anchor)
            camera_pose_solver.init_by_anchor(anchor)

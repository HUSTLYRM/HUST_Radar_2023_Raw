from queue import Queue

print('initializing...')
import cv2

# from threading import Thread
"""from threading import Lock
from multiprocessing import Process"""
import os
# import sys
import csv
import time
import numpy as np
# from math import pi

from ultralytics import YOLO
import torch
from ruamel.yaml import YAML

# import tools
# import alarm
import my_serial as messager
from stereo_camera import binocular_camera as bc
from stereo_camera.coex_matcher import CoExMatcher
# from apriltag_locater.apriltag_detector import ApriltagDetector
# import apriltag_locater.apriltag_detector as ad
# from apriltag_locater.hik_camera import camera as mc
from target import Targets

from anchor import Anchor
from anchor import set_by_hand
from macro import *
import coordinate_conversion as cc

import gui

print('[main] modules imported')

# import global_var as gv
# 敌方颜色：1红、2蓝
ENEMY_COLOR = RED
portx = 'COM3'

main_cfg_path = "./videotest_config.yaml"
binocular_camera_cfg_path = "./video/bin_cam_config.yaml"
monocular_camera_cfg_path = "./apriltag_locater/hik_camera/config.yaml"
video_test_cfg = YAML().load(open(main_cfg_path, encoding='Utf-8', mode='r'))
bin_cam_cfg = YAML().load(open(binocular_camera_cfg_path, encoding='Utf-8', mode='r'))
# _cam_cfg = YAML().load(open(binocular_camera_cfg_path, encoding='Utf-8', mode='r'))
device = torch.device('cuda:0')

if video_test_cfg['debug']:
    portx = 'COM7'

CarsTotal = 5
RedCarsID = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
BlueCarsID = {1: 101, 2: 102, 3: 103, 4: 104, 5: 105}
classes = ["car", "armor1red", "armor2red", "armor3red", "armor4red", "armor5red",
           "armor1blue", "armor2blue", "armor3blue", "armor4blue", "armor5blue", "base", "ignore"]

cos_arc_roll = np.cos(bin_cam_cfg['set']['roll'] * np.pi / 180)
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

    if video_test_cfg['training']:
        radar_base = [-60, 7000, 2500]

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
if video_test_cfg['ctrl']['SAVE_IMG']:
    time_now = time.localtime()
    img_folder = './record' + str(time_now[1]) + '-' + str(time_now[2]) + '-' + str(time_now[3]) + '-' + str(
        time_now[4])  # months + days + hours
    primal_folder = img_folder + '/primal'
    result_folder = img_folder + '/result'
    if not os.path.exists(img_folder):
        os.mkdir(img_folder)
        os.mkdir(primal_folder)
        os.mkdir(result_folder)

if video_test_cfg['ctrl']['RECORDING']:
    time_now = time.localtime()
    video_folder = './video_record' + str(time_now[1]) + '-' + str(time_now[2]) + '-' + str(time_now[3]) + '-' + str(
        time_now[4]) + '-' + str(time_now[5]) + '-' + str(time_now[6])  # months-days-hours-mins-secs
    raw_video_folder = video_folder + '/raw'
    left_video_folder = raw_video_folder + '/left'
    right_video_folder = raw_video_folder + '/right'
    # result_video_folder = video_folder + '/result'
    if not os.path.exists(video_folder):
        os.mkdir(video_folder)
    if not os.path.exists(raw_video_folder):
        os.mkdir(raw_video_folder)
    if not os.path.exists(left_video_folder):
        os.mkdir(left_video_folder)
    if not os.path.exists(right_video_folder):
        os.mkdir(right_video_folder)
        # os.mkdir(result_video_folder)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # ('M', 'P', '4', 'V')
    frame_size = (bin_cam_cfg['param']['Width'], bin_cam_cfg['param']['Height'])
    left_video = cv2.VideoWriter(left_video_folder + "/raw_left.mp4", fourcc, 12, frame_size, True)
    right_video = cv2.VideoWriter(right_video_folder + "/raw_right.mp4", fourcc, 12, frame_size, True)
    depth_video = cv2.VideoWriter(video_folder + "/dep_view_left.mp4", fourcc, 12, frame_size, True)
    disp_video = cv2.VideoWriter(video_folder + "/disp_left.mp4", fourcc, 12, frame_size, True)

if video_test_cfg['ctrl']['SAVE_CSV']:
    header = ['car_center_x', 'car_center_y', 'x', 'y', 'z']
    chart = open("data.csv", "w", newline='')
    writer = csv.DictWriter(chart, header)
    writer.writeheader()

print('preparing gui')

print('done')


# with qt
def push_button_clicked_quit():
    global exit_signal, Loop
    exit_signal = True
    print('exit')
    Loop = False


Loop = True


# 主函数
def main():
    global targets
    camera_left = camera_right = None
    ret_p = ret_q = None
    coex_matcher = None
    model_car = None  # = model_armor
    ir = suffix = None
    dst_img = None
    # send_queue = Queue()

    if video_test_cfg['ctrl']['MODE'] == 'video':
        camera_left, fpsl, sizel = bc.get_video_loader(video_test_cfg['video_left'])
        camera_right, fpsr, sizer = bc.get_video_loader(video_test_cfg['video_right'])

    elif video_test_cfg['ctrl']['MODE'] == 'camera':
        print("\nLoading camera")
        camera_left, camera_right, ret_p, ret_q = bc.get_camera(bin_cam_cfg)
        print("Done")

    print("\nLoading matching model")
    coex_matcher = CoExMatcher(bin_cam_cfg)
    print("Done\n")

    left_cam_cfg = dict()
    left_cam_cfg['intrinsic'] = bin_cam_cfg['calib']['intrinsic1']
    left_cam_cfg['distortion'] = bin_cam_cfg['calib']['distortion1']
    camera_pose_solver = cc.CameraPoseSolver(ALLY_COLOR, left_cam_cfg)
    if video_test_cfg['ctrl']['ANCHOR']:
        anchor = Anchor()
        while True:  # this while is for case where no img got
            ret, image_left = camera_left.read()
            ret, image_right = camera_right.read()

            if image_left is None:
                continue
            set_by_hand(image_left, anchor)
            camera_pose_solver.init_by_anchor(anchor)
            break

    if video_test_cfg['ctrl']['DETECT']:
        # 加载模型
        print('Loading Car Model')
        model_car = YOLO(video_test_cfg['weights']['yolov8'])
        # './weights/with_sod_yolov8l_epoch140_16_640_adam_half_0705_.pt'

        # as warmup
        # dummy = cv2.imread('./dummy.png')
        # model_car.predict(dummy)
        # stride = model_car.model.stride.numpy()
        # stride = 1  # 32
        # shape_shifter = tools.pre_cfg(cfg['hyper']['imgsz'], cfg['hyper']['image_h'], cfg['hyper']['image_w'], stride)
        print('Done\n')

    cnt = 0
    start = time.time()
    # now = time.time()
    last = time.time()
    # Here the main loop
    global Loop
    while Loop:
        if cv2.waitKey(1) == ord('q'):
            Loop = False

        ret, image_left = camera_left.read()
        ret, image_right = camera_right.read()
        if not ret:
            print("Done!")
            break
        if image_right is None or image_left is None:
            continue

        if video_test_cfg['ctrl']['RECORDING']:
            left_video.write(image_left)
            right_video.write(image_right)

        re_left, point_cloud, disp_np = coex_matcher.inference(image_left, image_right)
        if video_test_cfg['ctrl']['RECORDING']:
            disp_video.write(disp_np)
        disp_np = cv2.applyColorMap(2 * disp_np, cv2.COLORMAP_MAGMA)
        cv2.imshow('disp', disp_np)

        if video_test_cfg['ctrl']['RECORDING']:
            cnt += 1
            fps = cnt / (time.time() - start)
            cv2.putText(
                disp_np,
                "%.1f fps" % fps,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            depth_video.write(disp_np)

        if video_test_cfg['ctrl']['DETECT']:
            dst_img = np.copy(re_left)


            result = model_car.predict(dst_img, show=True)
            boxes = result[0].boxes.data.cpu()
            boxes = boxes.numpy()

            print(boxes)

            targets.update(boxes)
            # DONE: add transform TODO: test transform
            for target in targets.targets:
                if target.conf > 0:
                    cam_coord = [[point_cloud[int(target.center_yx[0])][int(target.center_yx[1])][0]],
                                 [point_cloud[int(target.center_yx[0])][int(target.center_yx[1])][1]],
                                 [point_cloud[int(target.center_yx[0])][int(target.center_yx[1])][2]]]
                    field_coord = camera_pose_solver.get_field_coord(cam_coord)
                    target.x = field_coord[0][0]
                    target.y = field_coord[1][0]
                    if video_test_cfg['debug'] or video_test_cfg['training']:
                        msg = str(cam_coord)
                        cv2.putText(re_left,
                                    msg,
                                    (int(target.center_yx[1]), int(target.center_yx[0])),
                                    cv2.FONT_HERSHEY_PLAIN,
                                    1.0,
                                    (0, 0, 255),
                                    thickness=1)
            if video_test_cfg['debug'] or video_test_cfg['training']:
                cv2.imshow('dist', re_left)
            # TODO: think about
            # cnt_send = 0
            for car in targets.targets:
                if car.conf > 0:
                    now = time.time()
                    # 距离上一次发送时间小于0.1s:sleep
                    if now - last < 0.1:
                        time.sleep(0.1 - (now - last))

                    print(car.get_id())
                    print(car.x / 1000)
                    print(car.y / 1000)
                    messager.send_enemy_location(ser, car.get_id(), car.x / 1000,
                                                 car.y / 1000)  # mm to m
                    last = time.time()

            """if cnt_send < 2:
                if now - last < 0.1:
                    time.sleep(0.1 - (now - last))"""
            # messager.send_random(ser, label, seed / 1000, buf / 1000)
            # cv2.waitKey(100)
            # messager.send_random(ser, label, seed, buf)
            # last = time.time()

    '- end of loop -----------------------------------------------------------------------------'

    if video_test_cfg['ctrl']['MODE'] == 'camera':
        # 关闭相机并销毁句柄
        bc.camera_close(camera_left)
        bc.camera_close(camera_right)
    # 等待相机线程执行完毕(释放摄像头)
    cv2.destroyAllWindows()
    # 停止录像，释放视频头
    if video_test_cfg['ctrl']['RECORDING']:
        os.system('copy ./cam_config.yaml ' + video_folder + '/related_cam_config.yaml')
        left_video.release()
        right_video.release()
        depth_video.release()
        disp_video.release()

    print('release!')


if __name__ == '__main__':
    main()

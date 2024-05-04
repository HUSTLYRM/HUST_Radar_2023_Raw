print('initializing...')
import cv2
# from threading import Thread
# from threading import Lock
# from multiprocessing import Process
import os
import shutil
import sys
import csv
import time
import numpy as np
import torch
from ruamel.yaml import YAML
from ultralytics import YOLO

import my_serial as messager
from stereo_camera import binocular_camera as bc
from hik_camera import camera as mc
from stereo_camera.coex_matcher import CoExMatcher
# from apriltag_locater.apriltag_detector import ApriltagDetector
# import apriltag_locater.apriltag_detector as ad


from target import Targets, Armor  # Car,
from anchor import Anchor
from anchor import set_by_hand
from macro import *
import coordinate_conversion as cc
import alarm
from radar_utils.chessboard_corner import find_chessboard_corners


# import ui.ui as visualize
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import gui


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

# TODO: check
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
    radar_base = [-987.55, 9018.02, 2500]  # red base
    cam_pos = [radar_base[0] - cam_bias[0], radar_base[1] + cam_bias[1], radar_base[2] + cam_bias[2]]
    sign = 1
    ALLY_COLOR = RED

    if main_cfg['training']:
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
    # TODO: check
    dst = video_folder + '/cfg'
    shutil.copy(binocular_camera_cfg_path, dst)
    shutil.copy(monocular_camera_cfg_path, dst)

    fourcc_colored = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # ('M', 'P', '4', 'V')
    frame_size = (bin_cam_cfg['param']['Width'], bin_cam_cfg['param']['Height'])
    left_video = cv2.VideoWriter(left_video_folder + "/raw_left.mp4", fourcc_colored, 12, frame_size, True)
    right_video = cv2.VideoWriter(right_video_folder + "/raw_right.mp4", fourcc_colored, 12, frame_size, True)
    depth_video = cv2.VideoWriter(video_folder + "/dep_view_left.mp4", fourcc_colored, 12, frame_size, True)
    lf_video = cv2.VideoWriter(lf_video_folder + "/lf.mp4", fourcc_colored, 12, frame_size, True)
    # TODO: check
    fourcc_grey = cv2.VideoWriter_fourcc('d', 'i', 'v', 'x')
    disp_video = cv2.VideoWriter(video_folder + "/disp_left.divx", fourcc_grey, 12, frame_size, True)
if main_cfg['ctrl']['SAVE_CSV']:
    header = ['car_center_x', 'car_center_y', 'x', 'y', 'z']
    chart = open("data.csv", "w", newline='')
    writer = csv.DictWriter(chart, header)
    writer.writeheader()


# with qt
def push_button_clicked_quit():
    global exit_signal, Loop
    exit_signal = True
    print('exit')
    Loop = False


def handle_highway_boxes(highway_scene, image, zone, boxes, highway_image):
    armors = []
    index = []
    last = time.time()
    print(boxes)
    for i, box in enumerate(boxes):
        if box[1] > 640:
            if ENEMY_COLOR == BLUE and 1 <= int(box[5]) <= 5:
                armors.append(Armor(bbox=[box[0], box[1], box[2], box[3]],
                                    cls=int(box[5]),
                                    conf=box[4]))
            elif ENEMY_COLOR == RED and 6 <= int(box[5]) <= 10:
                armors.append(Armor(bbox=[box[0], box[1], box[2], box[3]],
                                    cls=int(box[5]),
                                    conf=box[4]))
            # print(np.where(box))
            boxes = np.delete(boxes, i, axis=0)

    armors.sort(key=lambda x: x.conf, reverse=True)
    for armor in armors:
        i, flag = alarm.multi_zone_judgement(armor.center_xy, zone)
        if flag:
            index.append(i)
        if armor.conf > 0 and alarm.is_inside(armor.center_xy, zone):
            now = time.time()
            if now - last < 0.1:
                time.sleep(0.1 - (now - last))
            messager.send_enemy_location(ser, armor.get_id(), armor.x / 1000, armor.y / 1000)
            last = time.time()
    if len(index) > 0:
        alarm.Draw_Frame(highway_image, zone[0], [255, 255, 255], thickness=4)
    else:
        alarm.Draw_Frame(highway_image, zone[0], [0, 255, 0], thickness=1)

    if main_cfg['ctrl']['GUI']:
        resized = cv2.resize(highway_image, (400, 200))
        frame = QImage(resized, 400, 200, 400 * 3, QImage.Format_BGR888)
        highway_scene.clear()  # 先清空上次的残留
        pixel_map = QPixmap.fromImage(frame)
        highway_scene.addPixmap(pixel_map)


Loop = True


# 主函数
def main():
    global targets
    camera_left = camera_right = camera_lf = None
    ret_p = ret_q = None
    coex_matcher = None
    model_car = None
    zone = None

    main_scene = None
    highway_scene = None
    timeout_draw_zone = False

    if main_cfg['ctrl']['MODE'] == 'video':
        camera_left, fps, size = bc.get_video_loader(main_cfg['video'])

    elif main_cfg['ctrl']['MODE'] == 'camera':
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

    if main_cfg['ctrl']['HIGHWAY']:
        nodata_cnt = 0
        lf_soft_trigger = False
        if main_cfg['zone']['LOAD_ZONE']:
            # 考虑时间因素以及方便相机姿态调整, 使用文件读取
            zone = alarm.load_area('highway.txt')
            print('Zone Loaded: highway.txt')
        else:

            while True:  # this while is for case where no img got
                camera_lf.MV_CC_SetCommandValue("TriggerSoftware")
                image = mc.get_frame(camera_lf, 'camera', ret_p)

                if image is None:
                    nodata_cnt += 1
                    if nodata_cnt > 10:
                        # lf_soft_trigger = True
                        main_cfg['ctrl']['HIGHWAY'] = False
                        break
                    continue
                timeout_draw_zone, zone = alarm.Draw_Zone_new(image, time_limit=7)

                if main_cfg['zone']['SAVE_ZONE']:
                    # write_area 的输入为[[(),()]]结构
                    alarm.write_area('highway.txt', zone)
                    print('Zone Saved: highway.txt')

                break
        print(zone)

    left_cam_cfg = dict()
    left_cam_cfg['intrinsic'] = bin_cam_cfg['calib']['intrinsic1']
    left_cam_cfg['distortion'] = bin_cam_cfg['calib']['distortion1']
    camera_pose_solver = cc.CameraPoseSolver(ALLY_COLOR, left_cam_cfg)
    if main_cfg['ctrl']['ANCHOR']:
        anchor = Anchor()
        while True:  # this while is for case where no img got
            image_left = bc.get_frame(camera_left, 'left_camera', ret_p)
            # to keep synchronous
            image_right = bc.get_frame(camera_right, 'right_camera', ret_q)

            if image_left is None or image_right is None:
                continue
            set_by_hand(image_left, anchor)
            camera_pose_solver.init_by_anchor(anchor)
            break
    else:
        # camera_pose_solver.init_by_constant()
        pass

    if main_cfg['ctrl']['DETECT']:
        # 加载模型
        print('Loading Car Model')
        model_car = YOLO(main_cfg['weights']['yolov8'])
        # as warmup
        # dummy = cv2.imread('./dummy.png')
        # model_car.predict(dummy)
        print('Done\n')

    if main_cfg['ctrl']['GUI']:
        print('preparing gui')
        # if QT_VERSION >= QT_VERSION_CHECK(5, 6, 0)
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
        app = QApplication(sys.argv)

        MainWindow = QMainWindow()
        monitor = gui.Ui_Monitor()
        monitor.setupUi(MainWindow)
        monitor.pushButton_quit.clicked.connect(push_button_clicked_quit)
        main_scene = QGraphicsScene()
        monitor.main_frame_view.setScene(main_scene)
        monitor.main_frame_view.show()
        highway_scene = QGraphicsScene()
        monitor.highway_frame_view.setScene(highway_scene)
        monitor.highway_frame_view.show()
        # TODO: debug msg
        # TODO: map view
        MainWindow.show()
        print('done')

        # TODO: maybe finish this
        if timeout_draw_zone:
            monitor.debug_msg.append('zone initialize: timeout, using saved one\n')

    cnt = 0
    start = 0.
    last = time.time()
    # Here the main loop
    global Loop
    while Loop:
        image_highway = None

        if cv2.waitKey(1) == ord('q'):
            Loop = False
        image_left = bc.get_frame(camera_left, 'left_camera', ret_p)
        image_right = bc.get_frame(camera_right, 'right_camera', ret_q)

        if main_cfg['ctrl']['HIGHWAY']:
            # if lf_soft_trigger:
            #     camera_lf.MV_CC_SetCommandValue("TriggerSoftware")
            image_highway = bc.get_frame(camera_lf, 'lf_camera', ret_p)
            # if image_highway is None:
            #     nodata_cnt += 1
            #     if nodata_cnt > 10:
            #         lf_soft_trigger = True

        if image_right is not None and image_left is not None:
            if main_cfg['ctrl']['RECORDING']:
                left_video.write(image_left)
                right_video.write(image_right)

                if main_cfg['ctrl']['HIGHWAY'] and image_highway is not None:
                    lf_video.write(image_highway)

            re_left, point_cloud, disp_np = coex_matcher.inference(image_left, image_right)

            if main_cfg['ctrl']['RECORDING']:
                disp_video.write(disp_np)
                cv2.imshow('raw_disp', disp_np)

            disp_np = cv2.applyColorMap(2 * disp_np, cv2.COLORMAP_MAGMA)
            if main_cfg['ctrl']['RECORDING']:
                # TODO: check
                depth_video.write(disp_np)

            if main_cfg['ctrl']['CHESSBOARD']:
                pattern_size = (8, 6)
                corners, image_with_corners = find_chessboard_corners(re_left, pattern_size)
                # print(corners)
                if corners is not None:
                    cv2.putText(image_with_corners,
                                '[ ' + str(
                                    round(point_cloud[int(corners[0][0][1])][int(corners[0][0][0])][0], 2)) + ', '
                                + str(round(point_cloud[int(corners[0][0][1])][int(corners[0][0][0])][1], 2)) + ', '
                                + str(round(point_cloud[int(corners[0][0][1])][int(corners[0][0][0])][2], 2)) + ']',
                                (int(corners[0][0][0]), int(corners[0][0][1])),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.7,
                                color=(255, 255, 255),
                                thickness=1,
                                lineType=cv2.LINE_AA
                                )
                    cv2.imshow('xyz', image_with_corners)

            if cnt == 10:
                start = time.time()
            if cnt > 10:
                now = time.time()
                fps = (cnt - 10) / (now - start)
                cv2.putText(disp_np,
                            "fps: " + "%.2f" % fps,
                            (4, 40),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.9,
                            color=(255, 255, 255),
                            thickness=2,
                            lineType=cv2.LINE_AA
                            )
            cv2.imshow('disp', disp_np)

            if main_cfg['ctrl']['DETECT']:
                if main_cfg['ctrl']['HIGHWAY'] and image_highway is not None:
                    dst_img = cv2.vconcat([re_left, image_highway])
                else:
                    dst_img = np.copy(re_left)

                result = model_car.predict(dst_img, show=True)
                boxes = result[0].boxes.data.cpu()
                boxes = boxes.numpy()

                print(boxes)
                if main_cfg['ctrl']['HIGHWAY'] and image_highway is not None:
                    handle_highway_boxes(highway_scene, dst_img, zone, boxes, image_highway)

                targets.update(boxes)
                # DONE: add transform TODO: test transform
                for target in targets.targets:
                    if target.conf > 0:
                        cam_coord = [[point_cloud[int(target.center_yx[0])][int(target.center_yx[1])][0]],
                                     [point_cloud[int(target.center_yx[0])][int(target.center_yx[1])][1]],
                                     [point_cloud[int(target.center_yx[0])][int(target.center_yx[1])][2]]]
                        field_coord = camera_pose_solver.get_field_coord(cam_coord)
                        target.x = field_coord[0]
                        target.y = field_coord[1]
                        if main_cfg['debug'] or main_cfg['training']:
                            msg = str(cam_coord)
                            cv2.putText(re_left,
                                        msg,
                                        (int(target.center_yx[1]), int(target.center_yx[0])),
                                        cv2.FONT_HERSHEY_PLAIN,
                                        1.0,
                                        (0, 0, 255),
                                        thickness=1)
                if main_cfg['debug'] or main_cfg['training']:
                    cv2.imshow('dist', re_left)

                for car in targets.targets:
                    if car.conf > 0:
                        now = time.time()
                        # 距离上一次发送时间小于0.1s:sleep
                        if now - last < 0.1:
                            time.sleep(0.1 - (now - last))
                        messager.send_enemy_location(ser, car.get_id(), car.x / 1000,
                                                     car.y / 1000)  # mm to m
                        if main_cfg['ctrl']['GUI']:
                            monitor.debug_msg.append('at [' + str(now) + '] send: ' + str(car.get_id()) + ' ' + str(car.x / 1000) +
                                                     ' ' + str(car.y / 1000) + '\n')
                        last = time.time()
            else:
                cv2.imshow('re_left', re_left)

            if main_cfg['ctrl']['GUI']:
                resized = cv2.resize(re_left, (720, 360))
                frame = QImage(resized, 720, 360, 720 * 3, QImage.Format_BGR888)
                main_scene.clear()  # 先清空上次的残留
                pixel_map = QPixmap.fromImage(frame)
                main_scene.addPixmap(pixel_map)

        cnt += 1
    '- end of loop -----------------------------------------------------------------------------'

    if main_cfg['ctrl']['MODE'] == 'camera':
        # 关闭相机并销毁句柄
        bc.camera_close(camera_left, 'camera_left')
        bc.camera_close(camera_right, 'camera_right')
        bc.camera_close(camera_lf, 'camera_lf')
    # 等待相机线程执行完毕(释放摄像头)
    cv2.destroyAllWindows()

    # 停止录像，释放视频头
    if main_cfg['ctrl']['RECORDING']:
        os.system('copy bin_cam_config.yaml ' + video_folder[2:] + '/cfg/bin_cam_config.yaml')
        os.system('copy mon_cam_config.yaml ' + video_folder[2:] + '/cfg/mon_cam_config.yaml')
        os.system('copy main_config.yaml ' + video_folder[2:] + '/cfg/main_config.yaml')
        left_video.release()
        right_video.release()
        disp_video.release()
        depth_video.release()
        lf_video.release()

    # guard_hot_line.join()
    if main_cfg['ctrl']['SAVE_CSV']:
        chart.close()
    print('release!')


"""def SerialSend():
    global ser, targets, exit_signal, ENEMY_COLOR

    last = time.time()
    while not exit_signal:
        now = time.time()
        for car in targets.targets:
            if car.conf > 0:
                now = time.time()
                # 距离上一次发送时间小于0.1s:sleep
                if now - last < 0.1:
                    time.sleep(0.1 - (now - last))
                messager.send_enemy_location(ser, car.get_id(ENEMY_COLOR), car.x / 1000, car.y / 1000)
                last = time.time()
                now = time.time()
        if now - last < 0.1:
            time.sleep(0.999 - (now - last))
        messager.send_random(ser, label, seed / 1000, buf / 1000)
        print('rand send')
        last = time.time()

    print('Serial Send Process Exit!')"""

"""def RandomSend():
    global ser, count_down, label, seed, buf
    time.sleep(44)  # wait for visual sheld
    last = time.time()
    while not exit_signal:
        now = time.time()
        # 距离上一次发送时间小于0.1s:sleep
        if now - last < 0.1:
            time.sleep(0.1 - (now - last))
        messager.send_random(ser, label, seed, buf)
        last = time.time()"""


def SerialReceive():
    global ser, count_down, exit_signal, blood, time_flag, blood_flag, blood_init
    blood_temp = [0 for _ in range(16)]
    blood_max = [0 for _ in range(16)]

    current_time_stamp = 420

    visible_list_total = np.zeros((5, 7), dtype=np.int8)

    def find_0xa5(_ser):
        last_bytes = 0
        while not exit_signal:
            current_bytes = _ser.in_waiting

            if last_bytes - current_bytes < -30:
                length = 30
                _info = _ser.read(length)

                for _index, number in enumerate(_info):
                    if number == 165:
                        if _index + 4 < length:
                            print('ok', _index, length)
                            _data_length = _info[_index + 1] + _info[_index + 2] * 256
                            seq = _info[_index + 3]
                            CRC8 = _info[_index + 4]

                            _header = messager.struct.pack('B', 165) + \
                                      messager.struct.pack('H', _data_length) + \
                                      messager.struct.pack('B', seq)

                            crc8 = messager.get_crc8_check_byte(_header)
                            if CRC8 == crc8:
                                print('success!')
                                return _index, _info[_index:]
                last_bytes = current_bytes

    index, info = find_0xa5(ser)
    data_length = info[1] + info[2] * 256

    # 保证 read 同步至一个帧尾
    while not exit_signal:
        if data_length + 9 - len(info) > 0:  # 未读完当前段
            ser.read(data_length + 9 - len(info))
            break
        else:  # 当前段全部位于info中
            info = info[data_length + 9:]
            if len(info) >= 3:
                data_length = info[1] + info[2] * 256
            else:
                ser.read()

    while not exit_signal:
        # print('ok')
        header = ser.read(5)
        while header == b'':
            time.sleep(0.01)

        data_length = header[1] + header[2] * 256
        frame_length = 4 + data_length
        frame_without_header = ser.read(frame_length)
        cmd_id = frame_without_header[0] + frame_without_header[1] * 256

        # 0201/0202/0203
        if cmd_id in [513, 514, 515, 516, 261]:
            continue

        else:
            # if cmd_id == 513:
            # print(frame_without_header)

            # 0001:时间
            if cmd_id == 1:
                game_state = (frame_without_header[2] & 240) // 16
                # 对战中
                if game_state == 4:
                    count_down = frame_without_header[3] + frame_without_header[4] * 256
                    time_flag = True
                    print("counting down: ", count_down)
                    if count_down == 0:
                        pass

            # 0003:血量
            elif cmd_id == 3:
                for i in range(16):
                    blood_temp[i] = frame_without_header[2 * i + 2] + frame_without_header[2 * i + 3] * 256
                    # 若升级或上场自动步兵, 更新最大血量
                    if blood_temp[i] > blood_max[i]:
                        blood_init = True
                        blood_max[i] = blood_temp[i]
                    # 出现不同时, 更新
                    if blood_temp[i] != blood[i]:
                        blood[i] = blood_temp[i]
                        blood_flag = True
                # print(blood_temp)
                # print(blood)

            # 0101:场地
            elif cmd_id == 257:
                pass

            # 0209:RFID
            elif cmd_id == 521:
                pass

            # 0301:自定义交互数据
            elif cmd_id == 769:
                data_id = frame_without_header[2] + frame_without_header[3] * 256
                # 自动围杀接收data_id:02FF
                if data_id == 767:
                    # 记录本次信息的发送方
                    sender_id = frame_without_header[4] + frame_without_header[5] * 256
                    receiver_id = frame_without_header[6] + frame_without_header[7] * 256
                    change_flag = frame_without_header[8]
                    visible = frame_without_header[9]
                    time_stamp = frame_without_header[10] + frame_without_header[11] * 26
                    # 利用最早的时间戳同步
                    if time_stamp < current_time_stamp:
                        current_time_stamp = time_stamp

                        """for index in range(len(available_list)):
                            available_list[index] = 0

                    index = car2index[sender_id]
                    available_list[index] = 1"""

                    # 可见装甲板数组, 低7位
                    visible_list = list(bin(visible)[bin(visible).find('b') + 1:])
                    # 补0
                    if len(visible_list) < 8:
                        for i in range(8 - len(visible_list)):
                            visible_list.insert(0, '0')
                    # 存入total数组
                    visible_list = np.array(visible_list).astype(np.int8)
                    visible_list_total[index] = visible_list[1:]
                    # visible_list中, 最后一位为英雄, 向前依次为工程/步兵...
                    print("current_stamp: ", current_time_stamp, " stamp: ", time_stamp, " sender: ", sender_id,
                          " visible: ", visible_list)

                    """if sum(available_list) >= Encircle_Trigger_num:
                        encirclement_target = Auto_encirclement(visible_list_total, blood, ENEMY_COLOR)
                    if encirclement_target != -1:
                        global_encirclement_flag = True
                        global_encirclement_target = encirclement_target"""
            # 其它, 例如比赛结束信息, 在个人服务器上测试时未出现
            else:
                print(hex(cmd_id), cmd_id)
                # print(frame_without_header)
    # print('Serial Receive Thread Exit!')

    # 1:0001, 时间截
    # 3:0003, 所有的血量
    # 257:0101, 场地事件
    # 261:0105, 飞镖发射倒计时
    # 516:0204, 增益
    # 521:0209, RFID : 基地增益、高地增益、能量机关激活点、飞坡增益、前哨站增益、补血点、复活卡


if __name__ == '__main__':
    # sender_process = Process(target=SerialSend())
    # sender_process.start()
    main()

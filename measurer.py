from coordinate_conversion import CameraPoseSolver
from point_picker import PointsPicker
from macro import *

print('initializing...')
import cv2
import os
import csv
import time
import numpy as np

import torch
from ruamel.yaml import YAML
from stereo_camera import binocular_camera as bc
from stereo_camera.coex_matcher import CoExMatcher

from anchor import Anchor, set_by_hand

print('[measurer] modules imported')

device = torch.device('cuda:0')

measurer_cfg_path = "./measurer_config.yaml"
binocular_camera_cfg_path = "./bin_cam_config.yaml"
monocular_camera_cfg_path = "./mon_cam_config.yaml"
measurer_cfg = YAML().load(open(measurer_cfg_path, encoding='Utf-8', mode='r'))
bin_cam_cfg = YAML().load(open(binocular_camera_cfg_path, encoding='Utf-8', mode='r'))
mon_cam_cfg = YAML().load(open(monocular_camera_cfg_path, encoding='Utf-8', mode='r'))

arc_roll = bin_cam_cfg['set']['roll'] * np.pi / 180
cos_arc_roll = np.cos(arc_roll)
cam_bias = bin_cam_cfg['set']['bias']

if measurer_cfg['ctrl']['RECORDING']:
    time_now = time.localtime()
    video_folder = './measure_video_record' + str(time_now[1]) + '-' + str(time_now[2]) + '-' + \
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

if measurer_cfg['ctrl']['SAVE_CSV']:
    header = ['car_center_x', 'car_center_y', 'x', 'y', 'z']
    chart = open("data.csv", "w", newline='')
    writer = csv.DictWriter(chart, header)
    writer.writeheader()

# 主函数
if __name__ == '__main__':

    print("\nLoading binocular camera")
    camera_left, camera_right, ret_p, ret_q = bc.get_camera(bin_cam_cfg)
    print("Done")

    print("\nLoading matching model")
    coex_matcher = CoExMatcher(bin_cam_cfg)
    print("Done\n")

    left_cam_cfg = dict()
    left_cam_cfg['intrinsic'] = bin_cam_cfg['calib']['intrinsic1']
    left_cam_cfg['distortion'] = bin_cam_cfg['calib']['distortion1']

    pp = PointsPicker()
    points = Anchor()
    solver = CameraPoseSolver(RED, left_cam_cfg)

    # anchor = Anchor()
    # while True:  # this while is for case where no img got
    #     image_left = bc.get_frame(camera_left, 'left_camera', ret_p)
    #     image_right = bc.get_frame(camera_right, 'right_camera', ret_q)
    #
    #     if image_left is None:
    #         continue
    #     set_by_hand(image_left, anchor)
    #     solver.init_by_anchor(anchor)
    #     break

    constant_file = './training_constant.yaml'
    constant_dict = YAML().load(open(constant_file, encoding='Utf-8', mode='r'))
    solver.init_by_constant(constant_dict)

    while True:  # this while is for case where no img got
        image_left = bc.get_frame(camera_left, 'left_camera', ret_p)
        image_right = bc.get_frame(camera_right, 'right_camera', ret_q)

        if image_right is not None and image_left is not None:
            if measurer_cfg['ctrl']['RECORDING']:
                left_video.write(image_left)
                right_video.write(image_right)

            re_left, point_cloud, disp_np = coex_matcher.inference(image_left, image_right)

            if measurer_cfg['ctrl']['RECORDING']:
                disp_video.write(disp_np)
                cv2.imshow('raw_disp', disp_np)

            disp_np = cv2.applyColorMap(2 * disp_np, cv2.COLORMAP_MAGMA)
            cv2.imshow('colored_disp', disp_np)

            pp.caller(re_left, points)

            while len(points) > 0:
                point = points.pop()

                cam_coord = [[point_cloud[int(point[1])][int(point[0])][0]],
                             [point_cloud[int(point[1])][int(point[0])][1]],
                             [point_cloud[int(point[1])][int(point[0])][2]]]

                # wld_coord = solver.get_field_coord(cam_coord)
                wld_coord = cam_coord # solver.get_field(cam_coord)
                print('wld_coord')
                print(wld_coord)

                msg = '[ ' + str(np.round(wld_coord[0][0], 2)) + ' , ' \
                      + str(np.round(wld_coord[1][0], 2)) + ' , ' + \
                      str(np.round(wld_coord[2][0], 2)) + ' ]'
                cv2.putText(re_left,
                            msg,
                            (int(point[0]), int(point[1]) + 40),
                            cv2.FONT_HERSHEY_PLAIN,
                            1.0,
                            (0, 0, 255),
                            thickness=1)

                pp.points_to_display.clear()

            cv2.imshow('re_left', re_left)
            key = cv2.waitKey()
            # if key ==

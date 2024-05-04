from audioop import cross
# from readline import parse_and_bind
from tools import *
from threading import Lock, Thread
from my_serial import *

import cv2
import numpy as np
import time
import math

# from main import Enemy_Cars


'''交互'''


def Draw_Point(event, x, y, flags, param):
    img = param[0]
    zone = param[1]
    if event == cv2.EVENT_LBUTTONDOWN:  # cv2.EVENT_LBUTTONDBLCLK 左键双击, 现改为左键按下
        print('(x:', x, ',y:', y, ')')
        str1 = '(x:' + str(x) + ',y:' + str(y) + ')'
        cv2.putText(img, str1, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=1)
        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
        zone.append((x, y))


def Draw_Frame(img, points, color, thickness):
    for i in range(len(points)):
        cv2.line(img, points[i - 1], points[i], color=color, thickness=thickness)


def Draw_Zone(img, Mode):
    multi_zone = []
    flag = True
    while flag:
        print('multi_zone', multi_zone)
        zone = []
        cv2.namedWindow('Setting')
        cv2.setMouseCallback('Setting', Draw_Point, (img, zone))

        if Mode == 'Alarm':

            while True:
                cv2.imshow('Setting', img)
                print('按照逆时针或顺时针框定区域, 且必须是凸图形。若图样非凸, 需用多个凸图形组合而成')
                key = cv2.waitKey(400)

                # 按下n(next) 下一个
                if key == ord('n'):
                    # 若图形不封闭则弃置
                    if len(zone) > 2:
                        multi_zone.append(zone)
                        Draw_Frame(img, zone, [0, 255, 0], thickness=1)  # 23.2.26update
                        # opencv新版本点坐标要求格式为元组(x,y), 故不再将列表转为np.array by cicecoo
                    break
                    # 按下q(quit) 退出
                elif key == ord('q'):
                    if len(zone) > 2:
                        multi_zone.append(zone)
                        Draw_Frame(img, zone, [0, 255, 0], thickness=1)
                        flag = False
                        cv2.imshow('Setting', img)
                        cv2.waitKey(1000)
                        cv2.destroyWindow('Setting')
                        print(multi_zone)
                        return multi_zone

                # 按下c(cancel)，取消设置
                elif key == ord('c'):
                    multi_zone = []
                    cv2.destroyWindow('Setting')
                    return multi_zone

        elif Mode == 'Effective':
            while True:
                cv2.imshow('Setting', img)
                print('按照逆时针或顺时针框定区域, 且必须是凸图形。')
                key = cv2.waitKey(400)

                # 按下q(quit) 退出
                if key == ord('q'):
                    if len(zone) > 2:
                        Draw_Frame(img, zone, [0, 255, 255], thickness=1)
                        flag = False
                        cv2.imshow('Setting', img)
                        cv2.waitKey(1000)
                        cv2.destroyWindow('Setting')
                        print(zone)
                        return zone

        else:
            return


def Draw_Zone_new(img, time_limit=30):
    multi_zone = []
    life_span = 7
    start = time.time()
    flag = True
    timeout = False
    while flag:
        print('multi_zone', multi_zone)
        zone = []
        cv2.namedWindow('Setting')
        cv2.setMouseCallback('Setting', Draw_Point, (img, zone))
        print('按照逆时针或顺时针框定区域, 且必须是凸图形。若图样非凸, 需用多个凸图形组合而成')
        while True:
            cv2.imshow('Setting', img)
            key = cv2.waitKey(4)
            # 按下n(next) 下一个
            if key == ord('n'):
                life_span = 7
                # 若图形不封闭则弃置
                if len(zone) > 2:
                    multi_zone.append(zone)
                    Draw_Frame(img, zone, [0, 255, 0], thickness=1)  # 23.2.26update
                    # opencv新版本点坐标要求格式为元组(x,y), 故不再将列表转为np.array by cicecoo
                break
                # 按下q(quit) 退出
            elif key == ord('q'):
                if len(zone) > 2:
                    multi_zone.append(zone)
                    Draw_Frame(img, zone, [0, 255, 0], thickness=1)
                    flag = False
                    cv2.imshow('Setting', img)
                    cv2.waitKey(1000)
                    cv2.destroyWindow('Setting')
                    print(multi_zone)
                    return timeout, multi_zone

            # 按下c(cancel)，取消设置
            elif key == ord('c'):
                cv2.destroyWindow('Setting')
                zone = load_area('highway.txt')
                return timeout, zone

            now = time.time()
            if now - start > time_limit:
                print('timeout')
                timeout = True
                zone = load_area('highway.txt')
                print('Zone Loaded: highway.txt')
                return timeout, zone


# 叉乘
def cross_product(p1, p2):
    return p1[0] * p2[1] - p1[1] * p2[0]


# 对于二维凸图形，判断某点是否恒在每条边的同一侧
# 注意: 要求顶点的顺序为顺时针或逆时针!!
def is_inside(point, zone):
    num = len(zone)
    pi = []
    node = []
    for i in range(num):
        pi.append((point[0] - zone[i][0], point[1] - zone[i][1]))

    for i in range(num):
        node.append(cross_product(pi[i - 1], pi[i]))
    return all(i >= 0 for i in node) or all(i <= 0 for i in node)


# 考虑到可能存在凸图形组合为非凸图形的情况 或 存在多个危险区域，对每个域的运算结果取或
def multi_zone_judgement(point, multi_zone):
    for index, zone in enumerate(multi_zone):
        if is_inside(point, zone):
            return index, True
    return 0, False


def Box_Deleting(boxes, Effective_Zone, shape_shifter):
    np_boxes = np.array(boxes[0].to('cpu'))

    Out_of_range_index = []
    for index, box in enumerate(np_boxes):

        # 坐标反变换后计算中心点坐标
        box = undo_resize(box, shape_shifter)
        center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

        if not is_inside(center, Effective_Zone):
            Out_of_range_index.append(index)

    np_boxes = np.delete(np_boxes, Out_of_range_index, axis=0)

    boxes = [torch.tensor(np_boxes).to('cpu')]  # 'cuda'

    return boxes


def zone_init(img=None):
    if img is None:
        multi_zone = [[(470, 620), (715, 415), (800, 350), (1150, 600)],
                      [(200, 450), (370, 630), (360, 640), (190, 630)]]

    else:
        multi_zone = [[]]

    return multi_zone


count = 0


def Alarm_detection(Enemy_Cars, Dangerous_Zone, Mode=2, Draw=False, dstimg=None):
    global count
    count += 1

    warn = False
    index = []

    # 使用世界坐标进行判断
    if Mode == 1:
        for Car in Enemy_Cars:
            i, alarm = multi_zone_judgement(Car.Get_Location(), Dangerous_Zone)
            if alarm:
                warn = True
                index.append(i)

    # 使用像素坐标进行判断
    elif Mode == 2:
        for Car in Enemy_Cars:
            # print(Car.Get_Location_picture())
            i, alarm = multi_zone_judgement(Car.Get_Location_picture(), Dangerous_Zone)
            if alarm:
                warn = True
                index.append(i)

    if Draw:
        for i, zone in enumerate(Dangerous_Zone):
            if i in index and ((count // 3) % 2 == 0) and warn:
                Draw_Frame(dstimg, zone, [255, 255, 255], thickness=4)
            else:
                Draw_Frame(dstimg, zone, [0, 255, 0], thickness=1)

    return warn


def Alarm(dstimg, Zone):
    pass

    # 在图像上显示红圈警告
    # cv2.circle(dstimg, (100,100), 30, color=(0,0,255), thickness=-1)

    # 若需要将警告发送至操作手界面，需要加入串口通信内容


if __name__ == '__main__':
    img = cv2.imread('3.jpg')
    multi_zone = Draw_Zone(img, "Alarm")

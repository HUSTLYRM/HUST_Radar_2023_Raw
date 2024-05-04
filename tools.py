import numpy as np
import math
import cv2
from numpy.core.fromnumeric import argmax, around
from my_serial import send_enemy_location
import torch
import time
import torchvision
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

# 车辆类，成员变量为车框在图像中的大小、颜色、id号、世界坐标

Car_Colorlist = {
    1: [255, 0, 0],  # 红英雄
    2: [255, 0, 255],  # 红工程
    3: [255, 140, 0],  # 红步兵3
    4: [255, 215, 0],  # 红步兵4
    5: [255, 193, 193],  # 红步兵5

    101: [0, 0, 255],  # 蓝英雄
    102: [175, 238, 238],  # 蓝工程
    103: [0, 255, 0],  # 蓝步兵3
    104: [105, 139, 34],  # 蓝步兵4
    105: [106, 90, 205]  # 蓝步兵5
}

Colorlist = {
    0: [255, 255, 255],
    1: [0, 0, 255],
    2: [255, 0, 0]
}

Car_Namelist = {
    1: '红英雄',
    2: '红工程',
    3: '红步兵3',
    4: '红步兵4',
    5: '红步兵5',

    101: '蓝英雄',
    102: '蓝工程',
    103: '蓝步兵3',
    104: '蓝步兵4',
    105: '蓝步兵5'
}

RedCarsID = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
BlueCarsID = {1: 101, 2: 102, 3: 103, 4: 104, 5: 105}


class Car:
    def __init__(self, x_min=0, y_min=0, x_max=0, y_max=0, ID=0, X=-1, Y=-1):
        # 车的ID
        self.ID = ID

        # 车在图像中的坐标：回归框的四个角点、框中心点
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

        self.center_x = 0
        self.center_y = 0

        # 车是否被定位到
        self.visible = False

        # 车的世界坐标X,Y
        self.X = X
        self.Y = Y

        self.depth = 0

        self.ttl = 0
        self.T = 200

    def Car_center(self):
        return (self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2

    # 若对车辆位置进行了更新，则设定其为可见
    def Update_Picture(self, xmin, ymin, xmax, ymax):
        center_x = (self.x_min + self.x_max) / 2
        center_y = (self.y_min + self.y_max) / 2

        order = center_x < self.x_min or center_x > self.x_max or center_y < self.y_min or center_y > self.y_max

        diff = math.sqrt((center_x - self.center_x) ** 2 + (center_y - self.center_y) ** 2)

        # if diff > self.T and self.ttl:
        #     print(diff)
        #     print(center_x,self.center_x,center_y,self.center_y)
        #     return

        if order and diff < self.T and self.ttl:
            return

        self.x_min = xmin
        self.y_min = ymin
        self.x_max = xmax
        self.y_max = ymax

        self.center_x = center_x
        self.center_y = center_y

        self.visible = True
        self.ttl = 5

    def Update_Location(self, X, Y):
        self.X = X
        self.Y = Y

    def Get_Location(self):
        return self.X, self.Y

    def Get_Location_picture(self):
        return int(self.center_x), int(self.center_y)

    def print(self):
        print('Car ID: ', self.ID, 'Car_coordinate: x:{0}, y:{1} '.format(self.X, self.Y), self.ttl)


class Armor():
    def __init__(self, xmin=0, ymin=0, xmax=0, ymax=0, ID=0):
        # 车的ID
        self.center = None
        self.ID = ID

        # 车在图像中的坐标：回归框的四个角点、框中心点
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.centerx = 0
        self.centery = 0

        # 车是否被定位到
        self.visible = False

        self.ttl = 0

    def get_center(self):
        return self.center

    # 若对车辆位置进行了更新，则设定其为可见
    def Update_Picture(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.centerx = (self.xmin + self.xmax) / 2
        self.centery = (self.ymin + self.ymax) / 2
        
        self.center = (self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2
        
        self.visible = True
        self.ttl = 5

    def Get_Location_picture(self):
        return (int(self.centerx), int(self.centery))

    def Update_Location(self, X, Y):
        self.X = X
        self.Y = Y

    def print(self):
        pass


# 欧拉角转为旋转矩阵
def Euler2RotationMatrix(theta):
    R_x = np.array([
        [1., 0., 0.],
        [0., math.cos(theta[2]), -math.sin(theta[2])],
        [0., math.sin(theta[2]), math.cos(theta[2])]
    ])

    R_y = np.array([
        [math.cos(theta[1]), 0., math.sin(theta[1])],
        [0., 1., 0.],
        [-math.sin(theta[1]), 0., math.cos(theta[1])]
    ])

    R_z = np.array([
        [math.cos(theta[0]), -math.sin(theta[0]), 0.],
        [math.sin(theta[0]), math.cos(theta[0]), 0.],
        [0., 0., 1.]
    ])

    Rotation = np.around(np.matmul(np.matmul(R_z, R_x), R_y), decimals=2)
    return Rotation


Map_X_max = 27.2
Map_Y_max = 14.1


def GetWorldCoord(point_cloud, Enemy_Cars, img_size, Rotation, Translation):
    x_bias = 5
    y_bias = 5

    if Enemy_Cars[0].ID > 100:
        enemy = 'BLUE'
    else:
        enemy = 'RED'

    for car in Enemy_Cars:
        if car.ttl != 0:

            if x_bias < car.centerx < img_size[1] - x_bias and y_bias < car.centery < img_size[0] - y_bias:
                Location = point_cloud[int(car.centery) - y_bias:int(car.centery) + y_bias,
                                       int(car.centerx) - x_bias:int(car.centerx) + x_bias, :]
                Xmedian = []
                Ymedian = []
                Zmedian = []

                finite = np.isfinite(Location)
                for i in range(2 * x_bias):
                    for j in range(2 * y_bias):
                        if False not in finite[i, j, :]:
                            Xmedian.append(Location[i, j, 0])
                            Ymedian.append(Location[i, j, 1])
                            Zmedian.append(Location[i, j, 2])
                X = np.median(np.array(Xmedian))
                Y = np.median(np.array(Ymedian))
                Z = np.median(np.array(Zmedian)) * -1

                # coord为1*3向量，Rotation为3*3向量，Translation为1*3向量
                coord = np.matrix([X, Y, Z])
                after = np.matmul(coord, Rotation) + Translation
                # print('1',coord)
                # print('2',after)
                # 取邻域内有效值中位数作为距离
                car.X = after[0, 0]
                car.Y = after[0, 1]
                print(car.ID, car.X, car.Y)
            else:
                coord = np.matrix(point_cloud[int(car.centerx), int(car.centery), :3])
                coord[0, 2] *= -1

                after = np.matmul(coord, Rotation) + Translation
                car.X = after[0, 0]
                car.Y = after[0, 1]

        if enemy == 'BLUE':
            car.X = Map_X_max - car.X
            car.Y = Map_Y_max - car.Y
        else:
            car.X = -1
            car.Y = -1


# 图像预处理
def pre_img(im, shifter, new_shape=(640, 640), color=(114, 114, 114), stride=32, device=torch.device('cpu')):
    r, new_unpad, top, bottom, left, right = shifter

    img = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    # 转置、连续化、转移至GPU/CPU，类型为float
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float()
    img = img / 255  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch di
    # print(top,bottom,left,right)

    return img


# 画图
def plot_boxes_cv2(img, boxes, shifter, enemy_color, class_names=None):
    if class_names is None:
        class_names = ['car', 'red_armor', 'blue_armor']
    r = shifter[0]
    top = shifter[2]
    left = shifter[4]

    for i in range(len(boxes)):
        box = np.array(boxes[i])
        cls_id = int(box[5])

        # car
        if cls_id == 3 - enemy_color:
            continue

        x1 = int((box[0] - left) / r)
        y1 = int((box[1] - top) / r)
        x2 = int((box[2] - left) / r)
        y2 = int((box[3] - top) / r)

        rgb = Colorlist.get(cls_id, [255, 255, 255])
        img = cv2.putText(img, class_names[cls_id], (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8, rgb, 1)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 1)

    return img


# 画图 v2.0 未使用
def plot_boxes_cv2_V2(img, Cars, shifter, savename=None, color=None):
    img = np.copy(img)

    width = img.shape[1]
    height = img.shape[0]

    r = shifter[0]
    top = shifter[2]
    left = shifter[4]

    for Car in Cars:

        if Car.visible:

            # x1 = int(Car.xmin * width)
            # y1 = int(Car.ymin * height)
            # x2 = int(Car.xmax * width)
            # y2 = int(Car.ymax * height)

            print('visible')

            x1 = int((Car.x_min - left) / r)
            y1 = int((Car.y_min - top) / r)
            x2 = int((Car.x_max - left) / r)
            y2 = int((Car.y_max - top) / r)

            if color:
                rgb = color
            else:
                rgb = Car_Colorlist.get(Car.ID, [255, 255, 255])

            img = cv2.putText(img, 'car', (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8, rgb, 1)
            img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 1)

    if savename:
        print("save plot results to %s" % savename)
        cv2.imwrite(savename, img)

    return img


# 将标注好的车辆ID显示在车辆识别框中
def Draw_Car(dstimg, Enemy_Cars):
    for car in Enemy_Cars:
        # print(Car.Get_Location_picture())
        if car.ttl:
            cv2.circle(dstimg, car.Get_Location_picture(), radius=4, color=(255, 0, 255), thickness=-1)
            cv2.putText(dstimg, str(car.ID), (car.Get_Location_picture()[0] + 3, car.Get_Location_picture()[1] - 3),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.6, color=(255, 0, 255))


def Draw_Armor(dstimg, Enemy_Armors):
    for armor in Enemy_Armors:
        # print(Car.Get_Location_picture())
        if armor.ttl:
            cv2.circle(dstimg, armor.Get_Location_picture(), radius=4, color=(0, 255, 255), thickness=-1)
            cv2.putText(dstimg,
                        str(armor.ID),
                        (armor.Get_Location_picture()[0] + 3, armor.Get_Location_picture()[1] - 3),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=0.6, color=(0, 255, 255))


def Draw_Depth(dstimg, Enemy_Cars):
    for Car in Enemy_Cars:
        # print(Car.Get_Location_picture())
        if Car.ttl:
            cv2.putText(dstimg, '{:.3f}'.format(Car.depth),
                        (Car.Get_Location_picture()[0] - 5, Car.Get_Location_picture()[1] + 5),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.8, color=(255, 255, 0))


def Draw_Depth(dstimg, Enemy_Cars):
    for Car in Enemy_Cars:
        # print(Car.Get_Location_picture())
        if Car.ttl:
            cv2.putText(dstimg, '{:.2f}'.format(Car.X),
                        (Car.Get_Location_picture()[0] - 5, Car.Get_Location_picture()[1] + 5),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.8, color=(255, 255, 0))
            cv2.putText(dstimg, '{:.2f}'.format(Car.X),
                        (Car.Get_Location_picture()[0] + 5, Car.Get_Location_picture()[1] + 5),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.8, color=(255, 255, 0))


# 图像逆变换
def undo_resize(box, shifter):
    r = shifter[0]
    top = shifter[2]
    left = shifter[4]

    x1 = int((box[0] - left) / r)
    y1 = int((box[1] - top) / r)
    x2 = int((box[2] - left) / r)
    y2 = int((box[3] - top) / r)

    return [x1, y1, x2, y2]


def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names


# 读取Area(有效区域、预警区域)
def load_area(namesfile):
    Area = []
    Areas = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()

        for line in lines:
            numbers = line.split()
            Area.clear()

            if len(numbers) <= 4 or len(numbers) % 2 != 0:
                print('Area Error')
                exit(1)

            for index in range(len(numbers) // 2):
                Area.append((int(numbers[2 * index]), int(numbers[2 * index + 1])))

            Areas.append(Area[:])

    return Areas


# 保存Area(有效区域、预警区域)
def write_area(namesfile, Areas):
    fp = open(namesfile, 'w+')

    for Area in Areas:
        line = []
        for point in Area:
            line.append(str(point[0]))
            line.append(' ')
            line.append(str(point[1]))
            line.append(' ')
        line.append('\n')

        fp.writelines(line)


# nms

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    # print(np.array(prediction.cpu()))
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates:list of bool, value is true when > conf_thres
    # print(np.array(xc.cpu()))
    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence
        # print(x)
        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        "./simkai.ttf", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


# 回归框变换
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


# iou计算
def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


# 图像变换参数计算

def pre_cfg(imgsz, height, width, stride):
    r = min(imgsz[0] / max(height, width), 1.0)  # rate

    new_unpad = int(round(width * r)), int(round(height * r))

    dw, dh = imgsz[1] - new_unpad[0], imgsz[0] - new_unpad[1]  # wh padding

    dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return [r, new_unpad, top, bottom, left, right]


RedCarsID = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
BlueCarsID = {1: 101, 2: 102, 3: 103, 4: 104, 5: 105}

# 装甲板匹配
"""def Armor_Classify(bounding_box, shifter, Enemy_Car_Name, Enemy_Armors, Confidence_Armors, Center_Armors,
                   ENEMY_COLOR, conf_thres=0.8):
    num = int(bounding_box[5])
    # 若识别得到的数字不在车辆列表中，
    if ENEMY_COLOR == 1:
        ID = RedCarsID[num + 1]
    elif ENEMY_COLOR == 2:
        ID = BlueCarsID[num + 1]

    if ID not in Enemy_Car_Name:
        return False
    
    confidence = bounding_box[4]
    # 置信度不满足要求的认为是无数字装甲板或无效装甲板，筛除
    if confidence < conf_thres:
        return False

    r = shifter[0]
    top = shifter[2]
    left = shifter[4]

    # TODO: optimise
    x1 = int((bounding_box[0] - left) / r)
    y1 = int((bounding_box[1] - top) / r)
    x2 = int((bounding_box[2] - left) / r)
    y2 = int((bounding_box[3] - top) / r)

    # 针对一个车多个装甲板的情况，选择置信度最高的装甲板进行更新
    # 该装甲板未被识别到，数组存0，进行更新
    if Confidence_Armors[num] == 0:
        # 将置信度存入 Confidence_Armors
        Confidence_Armors[num] = confidence

        # 将装甲板中心存入 Center_Armors
        armor_centerx = (bounding_box[0] + bounding_box[2]) / 2
        armor_centery = (bounding_box[1] + bounding_box[3]) / 2
        Center_Armors[num] = [armor_centerx, armor_centery]

        if ENEMY_COLOR == 1:
            # 得到该次识别中敌方装甲板所对应的数字序号，加1取索引得到ID号
            ID = RedCarsID[num + 1]
            # 更新敌方车辆列表中，对应车辆的成员变量：图片中的对应位置

        elif ENEMY_COLOR == 2:
            ID = BlueCarsID[num + 1]

        Enemy_Armors[num].Update_Picture(x1, y1, x2, y2)
        # print("Init!",ID,confidence)

    else:
        # 若上一次存入的置信度小于该次识别到的装甲板的置信度，进行更新
        if Confidence_Armors[num] < confidence:
            Confidence_Armors[num] = confidence

            armor_centerx = (bounding_box[0] + bounding_box[2]) / 2
            armor_centery = (bounding_box[1] + bounding_box[3]) / 2
            Center_Armors[num] = [armor_centerx, armor_centery]

            if ENEMY_COLOR == 1:
                # 得到该次识别中敌方装甲板所对应的数字序号，加1取索引得到ID号
                ID = RedCarsID[num + 1]
                # 更新敌方车辆列表中，对应车辆的成员变量：图片中的对应位置

            elif ENEMY_COLOR == 2:
                ID = BlueCarsID[num + 1]

            Enemy_Armors[num].Update_Picture(x1, y1, x2, y2)
            # print("Update!",ID,confidence)

        else:
            return False
    return True"""


def Armor_Classify(img, size, shifter, Enemy_Car_Name, bounding_box, Enemy_Armors, Confidence_Armors, Center_Armors,
                   model, ENEMY_COLOR, yolo='yolov5', conf_thres=0.8):
    image_h, image_w = size

    r = shifter[0]
    top = shifter[2]
    left = shifter[4]

    x1 = int((bounding_box[0] - left) / r)
    y1 = int((bounding_box[1] - top) / r)
    x2 = int((bounding_box[2] - left) / r)
    y2 = int((bounding_box[3] - top) / r)

    # print(int(bounding_box[1]/imgsz[1]*image_w) , int(bounding_box[3]/imgsz[1]*image_w),
    # int(bounding_box[0]/imgsz[1]*image_h) , int(bounding_box[2]/imgsz[1]*image_h))
    # print(x1,x2,y1,y2)
    # 抠图:注意第一维为纵坐标，第二维为横坐标！！

    if yolo == 'yolov4':
        armor_img = img[int(bounding_box[1] * image_w): int(bounding_box[3] * image_w),
                    int(bounding_box[0] * image_h): int(bounding_box[2] * image_h)]
    elif yolo == 'yolov5':
        armor_img = img[y1:y2, x1:x2]

    if min(x1, x2, y1, y2) < 0 or max(x1, x2) > image_w or max(y1, y2) > image_h:
        return False

    armor_img = cv2.resize(armor_img, (32, 32))
    # cv2.imshow('',cv2.resize(armor_img,(512,512)))

    # cv2.imshow('',np.array(armor_img ))
    cv2.waitKey(1)

    # Armor Classifier Here
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])
    tensor = transform(armor_img)
    tensor = tensor.unsqueeze(0)

    input_data = torch.autograd.Variable(tensor, requires_grad=False)
    output_data = model(input_data)

    # softmax
    # ps = torch.exp(output_data) / torch.sum(torch.exp(output_data))
    # print(output_data,ps)

    # 使用num与confidence保存所识别得到的编号与置信度
    num = np.argmax(np.array(output_data))
    confidence = np.exp(np.max(np.array(output_data)))

    # 若识别得到的数字不在车辆列表中，
    if ENEMY_COLOR == 1:
        ID = RedCarsID[num + 1]
    elif ENEMY_COLOR == 2:
        ID = BlueCarsID[num + 1]

    if ID not in Enemy_Car_Name:
        return False

    # 置信度不满足要求的认为是无数字装甲板或无效装甲板，筛除
    if confidence < conf_thres:
        return False

    # 针对一个车多个装甲板的情况，选择置信度最高的装甲板进行更新

    # 该装甲板未被识别到，数组存0，进行更新
    if Confidence_Armors[num] == 0:

        # 将置信度存入 Confidence_Armors
        Confidence_Armors[num] = confidence

        # 将装甲板中心存入 Center_Armors
        armor_centerx = (bounding_box[0] + bounding_box[2]) / 2
        armor_centery = (bounding_box[1] + bounding_box[3]) / 2
        Center_Armors[num] = [armor_centerx, armor_centery]

        if ENEMY_COLOR == 1:
            # 得到该次识别中敌方装甲板所对应的数字序号，加1取索引得到ID号
            ID = RedCarsID[num + 1]
            # 更新敌方车辆列表中，对应车辆的成员变量：图片中的对应位置

        elif ENEMY_COLOR == 2:
            ID = BlueCarsID[num + 1]

        Enemy_Armors[num].Update_Picture(x1, y1, x2, y2)
        # print("Init!",ID,confidence)

    else:
        # 若上一次存入的置信度小于该次识别到的装甲板的置信度，进行更新
        if Confidence_Armors[num] < confidence:

            Confidence_Armors[num] = confidence

            armor_centerx = (bounding_box[0] + bounding_box[2]) / 2
            armor_centery = (bounding_box[1] + bounding_box[3]) / 2
            Center_Armors[num] = [armor_centerx, armor_centery]

            if ENEMY_COLOR == 1:
                # 得到该次识别中敌方装甲板所对应的数字序号，加1取索引得到ID号
                ID = RedCarsID[num + 1]
                # 更新敌方车辆列表中，对应车辆的成员变量：图片中的对应位置

            elif ENEMY_COLOR == 2:
                ID = BlueCarsID[num + 1]

            Enemy_Armors[num].Update_Picture(x1, y1, x2, y2)
            # print("Update!",ID,confidence)

        else:
            return False
    return True


def Read_hex(string):
    for _ in string:
        print(hex(_), end=' ')
    print('\n')


# 发送地图信息

def Map(Enemy_Cars, ser):
    for Car in Enemy_Cars:
        if Car.visible == False:  # 非敌方机器人或未获取到点云信息的机器人
            continue

        # plt.scatter(car.X, car.Y, s=area, c=color, alpha=0.5, label='Enemy Car')

        send_enemy_location(ser, Car.ID, Car.X, Car.Y)

        # plt.show()
        # time.sleep()
        # plt.close('all')

import pupil_apriltags as apriltag
import cv2
import numpy as np
from ruamel.yaml import YAML
from scipy.spatial.transform import Rotation as R

# from hik_camera.params import *

info = False
chessboard = False
normal_line = False


class ApriltagDetector:
    def __init__(self):
        self.detector = apriltag.Detector(families='tag36h11')
        self.cfg = YAML().load(open('apriltag_locater/hik_camera/mon_cam_config.yaml', 'r'))

    def detect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tags = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=self.cfg['tag']['params'],
            tag_size=self.cfg['tag']['size']
        )
        if info:
            show = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            visualize(show, tags, "test")
        return tags


def get_cam_coord(tag):
    tag_coord = np.array([[tag.center_yx[0]], [tag.center_yx[1]], [0]])
    cam_coord = np.matmul(tag.pose_R, tag_coord)  # np.transpose(tag.pose_R)
    cam_coord += tag.pose_t
    """print()
    cv2.putText(
        img,
        str(cam_coord),
        org=(
            tag.corners[0, 0].astype(int) + 10,
            tag.corners[0, 1].astype(int) + 10,
        ),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=0.6,
        color=(200, 0, 175)
    )"""
    return cam_coord


def visualize(img, tags, name, dist=None):
    cfg = YAML().load(open('hik_camera/mon_cam_config.yaml', 'r'))
    angledim = []
    xyzstr = 'xyz'

    cv2.putText(
        img,
        str("%d apriltags have been detected." % len(tags)),
        org=(15, 15),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=0.6,
        color=(200, 0, 175)
    )

    for tag in tags:
        cv2.putText(
            img,
            str(tag.tag_id),
            org=(
                tag.corners[0, 0].astype(int) + 10,
                tag.corners[0, 1].astype(int) + 10,
            ),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=0.6,
            color=(0, 0, 255),
        )
        homo = tags[0].homography
        num, Rs, Ts, Ns = cv2.decomposeHomographyMat(homo, np.array(cfg['calib']['intrinsic']))

        r = R.from_matrix(Rs[1].T)
        eulerangle = r.as_euler(xyzstr).T * 180 / np.pi
        angledim.append(eulerangle[2])

        for i in range(4):
            c = tags[0].corners[i]
            cv2.circle(img, tuple(c.astype(int)), 4, (255, 0, 0), 2)
        cv2.circle(img, tuple(tags[0].center_yx.astype(int)), 4, (18, 200, 20), 2)

        dirangle = (eulerangle[2] - 5) * np.pi / 180 * 1.8

        ARROW_LENGTH = 120
        deltax = np.sin(dirangle) * ARROW_LENGTH
        deltay = ARROW_LENGTH / 2 * np.cos(dirangle)
        newcenter = tags[0].center_yx + np.array([deltax, deltay])

        cv2.circle(img, tuple(newcenter.astype(int)), 8, (255, 0, 0), 5)
        cv2.line(img, tuple(newcenter.astype(int)), tuple(tags[0].center_yx.astype(int)), (255, 0, 0), 2)

    cv2.imshow(name, img)





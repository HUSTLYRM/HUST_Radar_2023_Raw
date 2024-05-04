import os
import cv2
import time

from hik_camera import camera
from apriltag_detector import ApriltagDetector

CALIB = False  # True  # False
if __name__ == '__main__':
    cam, ret_p = camera.get_camera(cfg_path='hik_camera/mon_cam_config.yaml')
    det = ApriltagDetector()

    if CALIB:
        title = format("calib_w%d_h%d_bx%d_by%d_@%d" % (1280, 1024, 0, 0, time.time()))
        ir = "./calib/{}".format(title)
        if not os.path.exists(ir):
            os.mkdir(ir)
        suffix = ".png"

    cnt = 0
    # Here the main loop
    Loop = True
    while Loop:
        if cv2.waitKey(1) == ord('q'):
            Loop = False

        cam.MV_CC_SetCommandValue("TriggerSoftware")
        image = camera.get_frame(cam, '0963', ret_p)
        if image is None:
            continue

        det.detect(image)
        cv2.imshow('img', image)

        if CALIB:
            cv2.waitKey(1500)
            if ret_p.contents:
                cv2.imwrite(ir + '/' + str(cnt) + suffix, image)

        cnt += 1

    # 关闭相机并销毁句柄
    camera.camera_close(cam)

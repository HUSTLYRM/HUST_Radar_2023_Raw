import os
import cv2
import time
from ruamel.yaml import YAML
from hik_camera import camera

monocular_camera_cfg_path = "mon_cam_config.yaml"
mon_cam_cfg = YAML().load(open(monocular_camera_cfg_path, encoding='Utf-8', mode='r'))

CALIB = True
trigger_soft = True


if __name__ == '__main__':
    cam, ret_p = camera.get_camera(mon_cam_cfg)
    # det = ApriltagDetector()
    if CALIB:
        title = format("calib_w%d_h%d_bx%d_by%d_@%d" % (1280, 1024, 0, 0, time.time()))
        ir = "./long_focal_calib/{}".format(title)

        title = format("mvs_w%d_h%d_bx%d_by%d_@%d" % (1280, 1024, 0, 0, time.time()))
        ir = "./mon_cam_mvs/{}".format(title)

        
        if not os.path.exists(ir):
            os.mkdir(ir)
        suffix = ".png"

    cnt = 0
    # Here the main loop
    Loop = True
    while Loop:
        if cv2.waitKey(1) == ord('q'):
            Loop = False
            break

        if trigger_soft:
            cam.MV_CC_SetCommandValue("TriggerSoftware")
        image = camera.get_frame(cam, '0963', ret_p)
        if image is None:
            continue

        # det.detect(image)
        cv2.imshow('img', image)

        if CALIB:
            cv2.waitKey(2000)
            if ret_p.contents:
                cv2.imwrite(ir + '/' + str(cnt) + suffix, image)

        cnt += 1

    # 关闭相机并销毁句柄
    camera.camera_close(cam)
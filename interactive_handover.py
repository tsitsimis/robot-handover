import numpy as np
from naoqi import ALProxy, motion
from Robot import Robot
from RobotConstants import *
from Classifier import Classifier
import time
import cv2
import utils
import timeit


# extract features and train classifier
toy = "green"
toy_color = "green"
clf = Classifier("./samples/" + toy + "/", toy_color)
clf.get_train_features()
clf.fit()

# initialize the robot
ip_local = "192.168.1.18"
# ip_external = "192.168.0.121"
ip_external = "169.254.28.162"  # ethernet
nao = Robot(ip=ip_external, port=9559)
nao.crouch()
nao.init_head()
nao.init_arm()
nao.init_camera()
time.sleep(3)

nao.pose_head = nao.motion_proxy.getPosition(HEAD, motion.FRAME_TORSO, True)
nao.arm_pose = nao.motion_proxy.getPosition(R_ARM, motion.FRAME_TORSO, True)
tf_head = utils.pose2tf(nao.head_pose)
tf_arm = utils.pose2tf(nao.arm_pose)
tol = 0.85

# main loop
while True:
    frame = nao.cam2numpy()  # read current image
    found, contour, contours = utils.get_object_contour2(frame, toy_color)

    if nao.grasped_it and (not found):
        grasped_it = False
        nao.open_hand()

    if found:
        box = utils.contour2box(contour)  # bounding box
        features = cx, cy, orientation, width, height = utils.contour_features(contour)
        features = np.array(features)
        score = utils.detection_score(contour, width, height, contours)

        # visual servoing and hand pose adaptation
        if cv2.contourArea(contour) > 1000:
            turn_yaw, turn_pitch, head_yaw, head_pitch = \
                utils.head_controller(nao.motion_proxy, cx, cy, tol=tol)
        else:
            turn_yaw = False
            turn_pitch = False
            head_yaw = 0
            head_pitch = 0

        if turn_yaw or turn_pitch:
            # print("the head is turning")
            nao.head_rot = np.array([0, head_pitch, head_yaw])
            nao.head_transform1 = utils.posor2tf(nao.head_pos0, nao.head_rot)
            utils.set_tf(nao.motion_proxy, HEAD, utils.flat_tf(nao.head_transform1))  # head's visual servo
        else:
            head_yaw = nao.motion_proxy.getAngles(HEAD_YAW, True)[0]
            head_pitch = nao.motion_proxy.getAngles(HEAD_PITCH, True)[0]
            nao.head_rot = np.array([0, head_pitch, head_yaw])
            T_h1 = utils.posor2tf(nao.head_pos0, nao.head_rot)
            T_e1 = np.dot(np.dot(T_h1, np.linalg.inv(nao.head_transform0)), nao.hand_transform0)
            utils.set_tf(nao.motion_proxy, R_ARM, utils.flat_tf(T_e1))  # hand's pose adaptation

            # update values
            nao.head_pose = nao.motion_proxy.getPosition(HEAD, motion.FRAME_TORSO, True)
            nao.arm_pose = nao.motion_proxy.getPosition(R_ARM, motion.FRAME_TORSO, True)
            tf_head = utils.pose2tf(nao.head_pose)
            tf_arm = utils.pose2tf(nao.arm_pose)

        # predict handover quality
        text = ""
        if score >= 0.9:
            x = features
            x = np.reshape(x, (1, -1))
            y_pred = clf.predict(x)

            if y_pred == 0:                 # bad handover
                color = (0, 0, 255)
                text = "bad"
            elif y_pred == 50:              # average handover
                color = (255, 255, 255)
                text = "average"
            elif y_pred == 100:             # good handover
                color = (255, 0, 0)
                text = "good"
                nao.open_hand(close=True)
                nao.grasped_it = True
                print("handover successful")
                # break

        cv2.drawContours(frame, [box], 0, color, 2)  # draw the bounding box of the detected object
        cv2.circle(frame, (cx, cy), 5, color, thickness=3)
        cv2.putText(frame, text, (np.int(cx), np.int(cy)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # show frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF  # press q to exit
    if key == ord("q"):
        break

nao.video_proxy.unsubscribe(nao.subscriber)

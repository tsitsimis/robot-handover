import numpy as np
from Robot import Robot
from Classifier import Classifier
import time
import cv2
import utils


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

time.sleep(0)

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
        # history, is_steady, sx, sy = utils.is_object_steady(nao.history, cx, cy)

        if nao.grasped_it:
            break
            # x = np.reshape(features, (1, -1))
            # y_pred = clf.predict(x)
            # nao.predictions[nao.cnt_grasp] = y_pred
            # nao.cnt_grasp += 1
            # if nao.cnt_grasp >= 5:
            #     if np.sum(nao.predictions) < 4 * 100:
            #         nao.grasped_it = False
            #         nao.cnt_grasp = 0
            #         nao.predictions = np.zeros(5)
            #         nao.open_hand()
            #     else:
            #         break

        text = ""
        if score >= 0.9:
            x = features
            x = np.reshape(x, (1, -1))
            y_pred = clf.predict(x)         # predict handover quality

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

        cv2.drawContours(frame, [box], 0, color, 2)  # draw the bounding box of the detected object
        cv2.circle(frame, (cx, cy), 5, color, thickness=3)
        cv2.putText(frame, text, (np.int(cx), np.int(cy)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # show frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF  # press q to exit
    if key == ord("q"):
        break

nao.video_proxy.unsubscribe(nao.subscriber)

from utils import *


my_dir = "/home/theodore/Documents/ECE/diplomatiki/NAO/snapshots8/green/"
X, Y, img_num = get_train_features(my_dir)                         # get features for all images
c = np.mean(X[np.where(Y == 100), :], axis=1)                      # best grasping
gamma_o = 1e-4
C_o = 1
clf = svm.SVC(kernel="rbf", gamma=gamma_o, C=C_o, probability=True)  # train classifier
print(np.mean(model_selection.cross_val_score(clf, X, Y, cv=10)))
clf.fit(X, Y)

# connect to NAO
motion_proxy = ALProxy("ALMotion", IP, PORT)
posture_proxy = ALProxy("ALRobotPosture", IP, PORT)
video_proxy = ALProxy("ALVideoDevice", IP, PORT)

# INITIALIZATIONS
# initialize nao configuration
posture_proxy.goToPosture("Crouch", 0.5)
pose_head = motion_proxy.getPosition(HEAD, motion.FRAME_TORSO, True)

t_h0 = np.array([[pose_head[0]], [pose_head[1]], [pose_head[2]]])  # set initial head + end-effector transform
theta_h = np.array([0, 20, 0]) * almath.TO_RAD
T_h0 = pos2tf(t_h0, theta_h)
T_h1 = T_h0

t_e0 = INIT_HAND_POS
theta_e = INIT_HAND_ROT
T_e0 = pos2tf(t_e0, theta_e)

init_head(motion_proxy)  # move head
init_arm2(motion_proxy)  # move arm
motion_proxy.setAngles(R_HAND, HAND_OPEN, 0.2)  # open hand

#############
pose_head = motion_proxy.getPosition(HEAD, motion.FRAME_TORSO, True)
pose_arm = motion_proxy.getPosition(R_ARM, motion.FRAME_TORSO, True)
tf_head = pos2tf2(pose_head)
tf_arm = pos2tf2(pose_arm)
# print np.dot(np.linalg.inv(tf_head), tf_arm)
#############

# initialize camera
subscriber = init_cam(video_proxy)

# initialize object movement history
history = Queue.Queue(maxsize=5)

grasped_it = False
cnt_grasp = 0
score_vals = np.zeros(1)
y = np.zeros(5)
# tol = 0.85  ### probably useless

time.sleep(5)
# ################# fixed configuration #############
# move head
head_yaw = 5 * almath.TO_RAD
head_pitch = 20 * almath.TO_RAD
theta_h1 = np.array([0, head_pitch, head_yaw])
T_h1 = pos2tf(t_h0, theta_h1)
set_tf(motion_proxy, HEAD, flat_tf(T_h1))
exit()
time.sleep(5)
# move arm
head_yaw = motion_proxy.getAngles(HEAD_YAW, True)[0]
head_pitch = motion_proxy.getAngles(HEAD_PITCH, True)[0]
theta_h1 = np.array([0, head_pitch, head_yaw])
T_h1 = pos2tf(t_h0, theta_h1)
T_e1 = np.dot(np.dot(T_h1, np.linalg.inv(T_h0)), T_e0)
set_tf(motion_proxy, R_ARM, flat_tf(T_e1))
exit()
while True:
    frame = cam2numpy(video_proxy, subscriber)                          # read image from NAO camera
    found, contour, _, contours = get_object_contour2(frame, "green")  # object contour

    if grasped_it and (not found):
        grasped_it = False
        motion_proxy.setAngles(R_HAND, HAND_OPEN, 0.2)

    if found:
        box = contour2box(contour)  # bounding box
        features = cx, cy, orientation, width, height = contour_features(contour)
        features = np.array(features)
        score = detection_score(contour, width, height, contours)
        history, is_steady, sx, sy = is_object_steady(history, cx, cy)

        if grasped_it:
            x = np.reshape(features, (1, -1))
            Y_pred = clf.predict(x)
            y[cnt_grasp] = Y_pred
            cnt_grasp += 1
            if cnt_grasp >= 5:
                if np.sum(y) < 4 * 100:
                    grasped_it = False
                    cnt_grasp = 0
                    y = np.zeros(5)
                    motion_proxy.setAngles(R_HAND, HAND_OPEN, 0.2)
                else:
                    break

        if score >= 0.9:
            x = features
            x = np.reshape(x, (1, -1))
            Y_pred = clf.predict(x)         # predict grasping quality

            if Y_pred == 0:                 # bad handover
                color = (0, 0, 255)
                text = "bad"
            elif Y_pred == 50:              # average handover
                color = (255, 255, 255)
                text = "average"
            elif Y_pred == 100:             # good handover
                color = (255, 0, 0)
                text = "good"
                motion_proxy.setAngles(R_HAND, HAND_CLOSE, 0.2)
                grasped_it = True

        cv2.drawContours(frame, [box], 0, color, 2)  # draw detected object bounding box
        cv2.circle(frame, (cx, cy), 5, color, thickness=3)
        cv2.putText(frame, text, (np.int(cx), np.int(cy)), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                              1, color, 2)

    # cv2.rectangle(frame, (np.int(640*tol), np.int(480*tol)),
    #               (640 - np.int(640*tol), 480 - np.int(480*tol)), color=(0, 0, 0), thickness=1)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # show frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF  # press q to exit
    if key == ord("q"):
        break

video_proxy.unsubscribe(subscriber)

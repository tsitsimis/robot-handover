import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from naoqi import ALProxy
from naoqi import motion
import vision_definitions as vd
import Image
from sklearn import svm, metrics, model_selection, neighbors, tree
from scipy.stats import multivariate_normal
from scipy import interpolate
import almath
import time
import Queue
from RobotConstants import *


def init_arm(ip, port, speed=0.2):
    target = [0.18109862506389618, -0.053951337933540344, 0.12278126180171967,
              1.0500797033309937, -0.433413565158844, 0.31683531403541565]
    motion_proxy = ALProxy("ALMotion", ip, port)
    chain_name = "RArm"
    space = motion.FRAME_TORSO
    axis_mask = AXIS_MASK_X + AXIS_MASK_Y + AXIS_MASK_Z + \
                AXIS_MASK_WX + AXIS_MASK_WY + AXIS_MASK_WZ
    motion_proxy.setPosition(chain_name, space, target, speed, axis_mask)


def init_cam(video_proxy):
    cam_bottom = 0
    fps = 12
    try:
        video_proxy.unsubscribe("demo")
    except:
        pass
    subscriber = video_proxy.subscribeCamera("demo", cam_bottom, vd.kVGA, vd.kRGBColorSpace, fps)
    return subscriber


def cam2numpy(video_proxy, subscriber):
    # read image from NAO camera
    nao_image = video_proxy.getImageRemote(subscriber)
    image_width = nao_image[0]
    image_height = nao_image[1]
    array = nao_image[6]
    # convert NAO's image to numpy array
    frame = Image.frombytes("RGB", (image_width, image_height), array)
    frame = np.array(frame)
    return frame


def color_range(color):
    if color == "green":
        lower = (42, 42, 56)
        upper = (83, 255, 255)
    elif color == "cyan":
        lower = (0, 129, 95)
        upper = (255, 255, 255)
    elif color == "red":
        lower = (0, 146, 118)
        upper = (255, 255, 255)
    elif color == "marker":
        lower = (0, 92, 171)
        upper = (78, 255, 255)
    elif color == "red_cylinder":
        lower = (0, 155, 131)
        upper = (255, 255, 255)
        # lower = (155, 129, 149)
        # upper = (255, 255, 255)
    elif color == "blue_rectangle":
        lower = (28, 60, 78)
        upper = (131, 255, 255)
    elif color == "orange_cylinder":
        lower = (0, 67, 103)
        upper = (255, 255, 255)
    elif color == "yellow_stairs":
        lower = (0, 208, 52)
        upper = (255, 255, 255)
    elif color == "red_cube":
        lower = (0, 156, 66)
        upper = (255, 255, 255)

    return lower, upper


def get_object_contour(img, color):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # convert to HSV color space

    lower, upper = color_range(color)  # threshold min & max values
    thresh = cv2.inRange(hsv, lower, upper)  # apply threshold in HSV

    # morphological filtering
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

    # contours
    thresh = closing
    contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        return True, c, thresh
    return False, [], thresh


def contour2box(c):
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = box.astype(int)
    return box


def contour_features(contour):
    # fit rectangle
    ((_, _), (width, height), angle) = cv2.minAreaRect(contour)  # rectangle

    # centroid
    mom = cv2.moments(contour)
    cx = int(mom['m10'] / mom['m00'])
    cy = int(mom['m01'] / mom['m00'])

    # orientation
    if width > height:
        orientation = np.abs(angle)
    else:
        orientation = (90 - np.abs(angle))

    # rectangle corners
    nw_x = cx - width / 2
    nw_y = cy - height / 2

    ne_x = cx + width / 2
    ne_y = cy - height / 2

    sw_x = cx - width / 2
    sw_y = cy + height / 2

    se_x = cx + width / 2
    se_y = cy + height / 2

    # features vector
    features = [cx, cy, orientation, width, height]
    # features = [cx, cy, nw_x, nw_y, ne_x, ne_y, sw_x, sw_y, se_x, se_y, orientation, width, height]
    return features


def get_train_features(directory):
    files = os.listdir(directory)
    n_samples = len(files)
    n_features = 5
    # n_features = 13

    X = np.zeros((n_samples, n_features))
    Y = np.zeros(n_samples, dtype=int)
    img_num = np.zeros(n_samples)

    for i in range(0, n_samples):
        img_name = files[i]
        Y[i], img_num[i] = parse_filename(img_name)

        full_path = directory + img_name
        img = cv2.imread(full_path)
        found, contour, _ = get_object_contour(img, "green")
        if found:
            X[i, :] = contour_features(contour)

    return X, Y, img_num


def parse_filename(img_name):
    components = img_name.split("_")

    # get img number
    num = components[1]
    num = int(num)

    # get img score
    score = components[2]
    score = score.split('.')[0]
    # score = float(score) / 100
    score = np.int(score)

    return score, num


def create_train_test(features, p):
    n_samples = np.shape(features)[0]
    all_ind = range(0, n_samples)

    # train set
    n_train = np.ceil(p * n_samples)
    train_ind = np.random.choice(all_ind, n_train, replace=False)
    train_set = features[train_ind, :]

    # test set
    test_ind = [x for x in all_ind if x not in train_ind]
    test_set = features[test_ind, :]

    return train_ind, train_set, test_ind, test_set


def plot_decision_boundary(clf, X, Y):
    h = 0.5
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.coolwarm)

    # plt.xlabel('centroid x')
    # plt.ylabel('orientation')
    # plt.title('decision boundaries')
    #
    # classes = ['good', 'medium', 'bad']
    # class_colours = ['red', 'white', 'blue']
    # recs = []
    # for i in range(0,len(class_colours)):
    #     recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=class_colours[i]))
    # plt.legend(recs,classes,loc=4)


def is_object_steady(history, cx, cy):
    # add new position
    history.put([cx, cy])

    # convert to numpy array
    history_np = np.array(list(history.queue))

    # calc std of x, y coordinates
    sx = np.std(history_np[:, 0])
    sy = np.std(history_np[:, 1])

    # remove oldest position
    if history.full():
        history.get()

    # check if object is steady
    is_steady = (sx <= 30 and sy <= 30)

    return history, is_steady, sx, sy


def nao_response(bad_dim):
    if bad_dim == 0:
        return "Bring the object closer to the x axis"
    elif bad_dim == 1:
        return "Bring the object closer to the y axis"
    elif bad_dim == 2:
        return "Rotate the object"
    elif bad_dim == 3 or bad_dim == 4:
        return "Bring the object closer to the z axis"


def train_color_model():
    my_dir = "/home/theodore/Documents/ECE/diplomatiki/NAO/code/python/nao_handover/color_samples/txt/"
    files = os.listdir(my_dir)
    n_samples = len(files)

    samples = np.zeros((0, 3))

    for i in range(0, n_samples):
        full_path = my_dir + files[i]
        color_points = np.loadtxt(full_path)
        samples = np.concatenate((samples, color_points), axis=0)

    mean = np.mean(samples, axis=0)
    cov = np.cov(samples, rowvar=0)

    return mean, cov


def proba_image(img, mean, cov, d3=True):
    proba_img = multivariate_normal.pdf(img, mean, cov)
    proba_img /= np.max(proba_img)
    proba_img *= 255

    if d3:
        proba_img = np.dstack([proba_img] * 3)

    return proba_img


def nao_stiffness(ip, port, v):
    motion_proxy = ALProxy("ALMotion", ip, port)
    motion_proxy.setStiffnesses(["Body"], v)


def nao_posture(ip, port, posture, speed=0.5):
    posture_proxy = ALProxy("ALRobotPosture", ip, port)
    posture_proxy.goToPosture(posture, speed)  # blocking call


def get_hand_tf(ip, port):
    motion_proxy = ALProxy("ALMotion", ip, port)
    name = 'RArm'
    space = 0
    use_sensor_values = False
    tf = motion_proxy.getTransform(name, space, use_sensor_values)

    r_shoulder_roll = motion_proxy.getAngles(R_SHOULDER_ROLL, use_sensor_values)
    return r_shoulder_roll


def move_hand(ip, port, tf0):
    motion_proxy = ALProxy("ALMotion", ip, port)
    space = 0  # FRAME_TORSO
    axis_mask = 3
    is_absolute = False
    effector = "RArm"

    tf1 = tf0
    # tf1[7] += 0.02
    path = [tf1]
    path = [[1.0, 0.0, 0.0, 0.0,
             0.0, 1.0, 0.0, 0.05,
             0.0, 0.0, 1.0, 0.0,
             0.0, 0.0, 0.0, 1.0]]

    path = [0.0, 0.05, 0.0,
            0.0, 0.0, 0.0]

    duration = 2.0  # seconds
    # motion_proxy.transformInterpolation(effector, space, path, axis_mask, duration, is_absolute)
    # motion_proxy.positionInterpolation(effector, space, path, axis_mask, duration, is_absolute)
    motion_proxy.setAngles(R_SHOULDER_ROLL, 9.8 * almath.TO_RAD, 0.3)


def nao_move_shoulder(ip, port, delta_theta):
    motion_proxy = ALProxy("ALMotion", ip, port)

    theta0 = motion_proxy.getAngles(R_SHOULDER_ROLL, False)[0]
    theta1 = theta0 + delta_theta
    if theta1 > 17 * almath.TO_RAD:
        return False, theta1
    else:
        motion_proxy.setAngles(R_SHOULDER_ROLL, theta1, 0.03)
        theta1 = motion_proxy.getAngles(R_SHOULDER_ROLL, False)[0]
        return True, theta1


def nao_move_head(ip, port, delta_theta):
    motion_proxy = ALProxy("ALMotion", ip, port)

    theta0 = motion_proxy.getAngles(HEAD_YAW, False)[0]
    theta1 = theta0 + delta_theta
    motion_proxy.setAngles(HEAD_YAW, theta1, 0.03)
    theta1 = motion_proxy.getAngles(HEAD_YAW, False)[0]
    return theta1


def init_interpolation():
    data = np.array([[9.8, 0], [12, 2], [14, 3], [16, 6], [18, 7]])
    tck = interpolate.splrep(data[:, 0], data[:, 1], s=0)
    return tck


def shoulder2head(tck, theta_s):
    ynew = interpolate.splev(theta_s, tck, der=0)
    return ynew


def head_follow_marker(ip, port, x0, y0, cx, cy):
    motion_proxy = ALProxy("ALMotion", ip, port)
    head_speed = 0.03

    delta_x = (x0 - cx) / 640  # error
    if np.abs(delta_x) >= 0.0:
        K = 10
        delta_yaw = K * delta_x
        head_yaw = motion_proxy.getAngles(HEAD_YAW, False)[0] * almath.TO_DEG
        head_yaw += delta_yaw

        head_yaw = np.min([head_yaw, 70])
        head_yaw = np.max([head_yaw, -70])

        head_yaw = head_yaw * almath.TO_RAD
        motion_proxy.setAngles(HEAD_YAW, head_yaw, head_speed)

    # dy = (imageHeight / 2.0 - cy) / (imageHeight / 2)
    # if np.abs(dy) >= 0.2:
    #     K = 10
    #     deltaPicth = K * dy
    #     headPitch = motion_proxy.getAngles(headPitchName, False)[0] * almath.TO_DEG
    #     headPitch -= deltaPicth
    #
    #     headPitch = np.min([headPitch, 20])
    #     headPitch = np.max([headPitch, -30])
    #
    #     headPitchRad = headPitch * almath.TO_RAD
    #     motion_proxy.setAngles(headPitchName, headPitchRad, head_speed)


def get_object_contour2(img, color):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # convert to HSV color space

    lower, upper = color_range(color)  # threshold min & max values
    thresh = cv2.inRange(hsv, lower, upper)  # apply threshold in HSV

    # morphological filtering
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

    # contours
    thresh = closing
    contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        # return True, c, thresh, contours
        return True, c, contours
    # return False, [], thresh, []
    return False, [], []


def detection_score(contour, width, height, contours, w=0.3):
    # bbox score
    real_area = cv2.contourArea(contour)
    rect_area = width * height
    score_area = real_area / rect_area

    # noise score
    Nc = len(contours)
    areas_sum = 0
    for i in range(Nc):
        areas_sum += cv2.contourArea(contours[i])

    score_noise = real_area / areas_sum

    score = w * score_area + (1 - w) * score_noise
    return score_area, score_noise, score


def set_tf(motion_proxy, chain, tf, speed=0.5):
    space = motion.FRAME_TORSO
    motion_proxy.setTransform(chain, space, tf, speed, AXIS_MASK_ALL)


def euler2matrix(theta):
    R_x = np.array([[1,         0,                 0                ],
                    [0,         np.cos(theta[0]), -np.sin(theta[0]) ],
                    [0,         np.sin(theta[0]),  np.cos(theta[0]) ]
                    ])
    R_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])  ],
                    [0,                     1,    0                 ],
                    [-np.sin(theta[1]),   0,      np.cos(theta[1])  ]
                    ])
    R_z = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),    0],
                    [np.sin(theta[2]),     np.cos(theta[2]),    0],
                    [0,                     0,                  1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def pos2tf(t, theta):
    r = euler2matrix(theta)
    tf = np.concatenate((r, t), axis=1)
    tf = np.concatenate((tf, R0001), axis=0)
    return tf


def pos2tf2(pose):
    pose = np.array(pose)
    t = pose[0:3]
    theta = pose[3:6]
    t = np.array([[t[0]], [t[1]], [t[2]]])
    r = euler2matrix(theta)
    tf = np.concatenate((r, t), axis=1)
    tf = np.concatenate((tf, R0001), axis=0)
    return tf


def flat_tf(tf):
    return list(tf.flatten()[0:12])


def head_controller(motion_proxy, cx, cy, tol=0.2):
    head_yaw = motion_proxy.getAngles(HEAD_YAW, False)[0] * almath.TO_DEG
    head_pitch = motion_proxy.getAngles(HEAD_PITCH, False)[0] * almath.TO_DEG
    turn_yaw = False
    turn_pitch = False

    delta_x = (IMG_W / 2.0 - cx) / (IMG_W / 2)
    delta_y = (IMG_H / 2.0 - cy) / (IMG_H / 2)

    if np.abs(delta_x) >= tol:
        k = 10
        delta_yaw = k * delta_x
        head_yaw += delta_yaw
        head_yaw = np.min([head_yaw, 0])
        head_yaw = np.max([head_yaw, -10])
        turn_yaw = True

    if np.abs(delta_y) >= tol:
        k = 10
        delta_pitch = k * delta_y
        head_pitch -= delta_pitch
        head_pitch = np.min([head_pitch, 30])
        head_pitch = np.max([head_pitch, -5])
        turn_pitch = True

    head_yaw *= almath.TO_RAD
    head_pitch *= almath.TO_RAD

    return turn_yaw, turn_pitch, head_yaw, head_pitch


def init_head(motion_proxy):
    motion_proxy.setPosition(HEAD, motion.FRAME_TORSO, [0, 0, 0, 0, 20*almath.TO_RAD, 0], 0.5, AXIS_MASK_ALL)


def init_arm2(motion_proxy):
    motion_proxy.setPosition(R_ARM, motion.FRAME_TORSO,
                             list(np.concatenate((INIT_HAND_POS.flatten(), INIT_HAND_ROT))),
                             0.5, AXIS_MASK_ALL)


def motion_controller(motion_proxy, chain, pd):
    K = 0.5
    p = motion_proxy.getPosition(chain, motion.FRAME_TORSO, False)
    p = np.array(p)

    err = pd[0:3] - p[0:3]
    while np.max(np.abs(err)) > 0.02:
        p[0:3] += K * err
        motion_proxy.setPosition(chain, motion.FRAME_TORSO, list(p), 0.5, AXIS_MASK_ALL)

        p = motion_proxy.getPosition(chain, motion.FRAME_TORSO, False)
        p = np.array(p)
        err = pd[0:3] - p[0:3]
        print np.max(np.abs(err))


def get_counter(directory):
    files = os.listdir(directory)
    n_samples = len(files)
    if n_samples > 0:
        img_num = np.zeros(n_samples)

        for i in range(0, n_samples):
            img_name = files[i]
            _, img_num[i] = parse_filename(img_name)
        return int(np.max(img_num) + 1)
    return 1

import numpy as np
import os
import cv2


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

    # features vector
    features = [cx, cy, orientation, width, height]
    return features


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


def get_train_features(directory, color):
    files = os.listdir(directory)
    n_samples = len(files)
    n_features = 5
    # n_features = 13

    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=int)
    img_num = np.zeros(n_samples)

    for i in range(0, n_samples):
        img_name = files[i]
        y[i], img_num[i] = parse_filename(img_name)

        full_path = directory + img_name
        img = cv2.imread(full_path)
        found, contour, _ = get_object_contour(img, color)
        if found:
            X[i, :] = contour_features(contour)

    return X, y, img_num

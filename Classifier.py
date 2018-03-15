import os
import utils as uu
import numpy as np
import cv2
from sklearn import svm, model_selection
import Queue


class Classifier:
    def __init__(self, samples_dir, color):
        self.samples_dir = samples_dir
        self.color = color

        self.clf = None
        self.X = None
        self.y = None
        self.img_num = None
        self.c = None           # best grasping

        self.history = Queue.Queue(maxsize=5)

    def get_train_features(self):
        files = os.listdir(self.samples_dir)
        n_samples = len(files)
        n_features = 5

        X = np.zeros((n_samples, n_features))
        y = np.zeros(n_samples, dtype=int)
        img_num = np.zeros(n_samples)

        for i in range(n_samples):
            img_name = files[i]
            y[i], img_num[i] = uu.parse_filename(img_name)

            full_path = self.samples_dir + img_name
            img = cv2.imread(full_path)
            found, contour, _ = uu.get_object_contour(img, self.color)
            if found:
                X[i, :] = uu.contour_features(contour)

        self.X = X
        self.y = y
        self.img_num = img_num
        self.c = np.mean(X[np.where(self.y == 100), :], axis=1)

    def fit(self):
        gamma_o = 1e-4
        C_o = 1
        clf = svm.SVC(kernel="rbf", gamma=gamma_o, C=C_o, probability=True)
        clf.fit(self.X, self.y)
        self.clf = clf

    def predict(self, X):
        return self.clf.predict(X)

    def cross_val_score(self, k=10, mean=True):
        scores = model_selection.cross_val_score(self.clf, self.X, self.y, cv=k)
        if mean:
            return np.mean(scores)
        return scores

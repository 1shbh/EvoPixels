import cv2
import numpy as np
import math

class Canvas:
    def __init__(self, src_img, K, ds_fit=128):
        self.h, self.w, self.c = src_img.shape
        self.blank_arr = np.full((self.h, self.w, 3), 255, np.uint8)
        self.ds = ds_fit
        self.src_down = cv2.resize(src_img, (self.ds, self.ds),
                                   interpolation=cv2.INTER_AREA).astype(np.float32)
        blank_down = cv2.resize(self.blank_arr, (self.ds, self.ds),
                                interpolation=cv2.INTER_AREA).astype(np.float32)
        self.blank_MSE = float(np.mean((self.src_down - blank_down) ** 2))
        self.colors = self.get_colors(src_img, K).tolist()

    @staticmethod
    def get_colors(src_img, K):
        Z = src_img.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = min(K, max(1, len(Z)))
        _, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        return np.uint8(center)
























'file:   main.py'
'author: zhangge9194@pku.edu.cn'

import cv2
import os
import random
import numpy as np
from tqdm import tqdm
import time

IMG_DIR = 'assignment1/data/'
COMPRESSION = True
# file = random.choice(os.listdir(IMG_DIR))


def get_corners(file):
    w = 6
    h = 9
    img = cv2.imread(os.path.join(IMG_DIR, file))
    if COMPRESSION:
        img = cv2.resize(img, (960, 1280), interpolation=cv2.INTER_AREA)
    # print(img.shape)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    start = time.time()
    ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
    if ret:
        cv2.drawChessboardCorners(img, (h, w), corners, ret)
        cv2.imwrite(os.path.join('assignment1/output', file), img)

        print(corners.shape)

    return ret, corners, (img.shape[1], img.shape[0])


files = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]

objpoints = []  # 在世界坐标系中的三维点
imgpoints = []  # 在图像平面的二维点
for f in tqdm(files):
    w = 6
    h = 9

    objp = np.zeros((w * h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)

    ret, corners, img_size = get_corners(f)
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
    print(ret, f)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_size, None, None)

# 去畸变
img2 = cv2.imread('assignment1/data/IMG_7538.jpg')
if COMPRESSION:
    img2 = cv2.resize(img2, (960, 1280), interpolation=cv2.INTER_AREA)
h, w = img2.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
    mtx, dist, (w, h), 0, (w, h))  # 自由比例参数
dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)
cv2.imwrite('assignment1/output/corrected_IMG_7538.jpg', dst)

total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error += error
print("total error: ", total_error / len(objpoints))

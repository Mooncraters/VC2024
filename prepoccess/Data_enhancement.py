import cv2
import numpy as np
import os

def Rotate(image,angle = 15,scale = 0.9):
    w = image.shape[1]
    h = image.shape[0]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
    image = cv2.warpAffine(image, M, (w, h))
    return image

def Adjust_brightness(image, percentage):
    copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    for xi in range(0, w):
        for xj in range(0, h):
            copy[xj, xi, 0] = int(image[xj, xi, 0]*percentage)
            copy[xj, xi, 1] = int(image[xj, xi, 1]*percentage)
            copy[xj, xi, 2] = int(image[xj, xi, 2]*percentage)
    return copy

def Move(image, x, y):
    w = image.shape[1]
    h = image.shape[0]
    translation_matrix = np.float32([[1, 0, x], [0, 1, y]])
    aft = cv2.warpAffine(image, translation_matrix, (w, h))
    return aft

def SaltNoise(image, percentage = 0.05):
    Aft = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    Aft_Num = int(percentage * w * h)
    for i in range(0,Aft_Num):
        ranxi = np.random.randint(0, h - 1)
        ranxj = np.random.randint(0, w - 1)
        rani = np.random.randint(0, 3)
        if(np.random.randint(0, 1)==0):
            Aft[ranxi, ranxj, rani] = 0
        else:
            Aft[ranxi, ranxj, rani] = 255
    return Aft

def GaussianNoise(image, percentage = 0.05):
    Aft = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    Aft_Num = int(percentage * w * h)
    for i in range(0, Aft_Num):
        ranxi = np.random.randint(0, h - 1)
        ranxj = np.random.randint(0, w - 1)
        rani = np.random.randint(0, 3)
        Aft[ranxi, ranxj, rani] = np.random.randn(1)[0]
    return Aft
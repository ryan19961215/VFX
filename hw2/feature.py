#!/usr/bin/python
# -*- coding: UTF-8 -*-
import cv2
import math
import numpy as np
import scipy
import matplotlib.pyplot as plt
import copy

# main可以做的事情：計算infile的特徵點特徵點和特徵點向量（還有一些bug比如說128維向量有64維不見，不過暫時沒有問題），並且輸出輸出特徵點圖

# 輸入圖片檔（main程式會改用myio.py讀檔）
infile = 'mountain2.jpg'
# 輸出幫助辨識的圖
Blur = 'S0.jpg' #進行黑白並高斯模糊以後的圖片
Mimage = 'S1.jpg' #整張圖對M之反應
Margintreatment = 'S2.jpg' #進行邊緣響應消除後之M結果
outfile = 'S3.jpg' # 輸出Harris Detect的圖名

# 可調整變數
blursize = (9,9) # 高斯模糊變數，建議：(3,3) ~ (9,9)
blurvar = 3 # ：高斯模糊變數，建議：0 ~ 3
k = 0.04 # Harris估計特徵值公式變數，建議：0.04 ~ 0.15
bound = 10 # 消除邊緣響應，建議圖片愈大該數放愈大
pointcatch = 2 #取最多特徵點的最多數量，與下面的選比較不嚴格的
thresh_std = 0.75 #取特徵點的最差值，被乘數為最好的特徵值之值，與上面選比較不嚴格的
bit = 8 #SIFT規定


def find_sub_max(darr, n):
    arr = darr.reshape(-1)
    for i in range(n-1):
        arr_ = arr
        arr_[np.argmax(arr_)] = 0
        arr = arr_
    
    return np.max(arr_)



def feature_detect(colorimage):
    
    grayimage = cv2.cvtColor(colorimage , cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur( grayimage , blursize , blurvar)
    
    Iy , Ix = np.gradient(image)
    Ixx = Ix * Ix
    Ixy = Ix * Iy
    Iyy = Iy * Iy
    
    Sxx = cv2.GaussianBlur(Ixx, blursize , blurvar)
    Sxy = cv2.GaussianBlur(Ixy, blursize, blurvar)
    Syy = cv2.GaussianBlur(Iyy, blursize, blurvar)

    detA = (Sxx * Syy) - (Sxy * Sxy)
    traceA = Sxx + Syy
    
    M = detA - k * pow(traceA , 2)
    
    cv2.imwrite(Mimage, M)
    
    for i in range(M.shape[1]):
        for b in range(bound):
            M[b][i] = 0
            M[M.shape[0]-b-1][i] = 0
    for j in range(M.shape[0]):
        for b in range(bound):
            M[j][b] = 0
            M[j][M.shape[1]-b-1] = 0

    cv2.imwrite(Margintreatment, M)

    kernels = []
    for y in range(3):
        for x in range(3):
            if x == 1 and y == 1: continue
            ker = np.zeros((3, 3))
            ker[1, 1] = 1
            ker[y, x] = -1
            kernels.append(ker)

    Q = copy.copy(M)
    threshrank = find_sub_max(Q, pointcatch)
    threshratio = thresh_std * M.max()
    thresh = min( threshratio , threshrank)
    localMax = np.ones(M.shape)
    localMax[M < thresh] = 0

    for ker in kernels:
        d = np.sign(cv2.filter2D(M, -1, ker))
        d[d < 0] = 0
        localMax = d * localMax

    print('\n')
    print('There are', int(np.sum(localMax)),'corners')
    print('Threshold=' , thresh )
    feature_points = np.where(localMax > 0)
    print('Feature Detect Complete')

    return feature_points, Ix , Iy



def pointview(infile,feature_points):
    image = cv2.imread(infile)
    for i in range(len(feature_points[0])):
        cv2.circle( image , (feature_points[1][i],feature_points[0][i]) , 2 , [0,0,255] , -1 )
    
    return image



def vector4( fy, fx , ori_rotated):
    split_vec = []
    for b in range(bit):
        split_vec.append(np.sum(ori_rotated[b][fy:fy+4, fx:fx+4]))
    
    split_vec_set = []
    for i in split_vec:
        split_vec_1 = i / (np.sum(split_vec) + 1e-8)
        if split_vec_1 < 0.2:
            split_vec_set.append(split_vec_1)
        else:
            split_vec_set.append(0.2)

    sum_sv = (np.sum(split_vec_set) + 1e-8)
    split_vec_return = []
    for i in split_vec_set:
        split_vec_return.append( i / sum_sv )
        
    return split_vec_return



def vector12 (y , x , theta , orient):
    M = cv2.getRotationMatrix2D((12, 12), theta[y, x], 1)
    if y-12 < 0 or x-12 < 0:
        return 0
    ori_rotated = []
    for t in orient:
        rotate = cv2.warpAffine(t[y-12:y+12, x-12:x+12], M, (24, 24))
        ori_rotated.append(rotate)
    ori_rotated = np.array( ori_rotated)

    vector = []
    offsets = [4, 8, 12, 16]
    for fy in offsets:
        for fx in offsets:
            sub_vector = vector4(fy, fx, ori_rotated)
            vector.append(sub_vector)

    return vector



def feature_descripter(colorimage , fdt_image_value, Ix , Iy):

    fd = []

    Ixx = Ix * Ix
    Iyy = Iy * Iy
    
    
    Igm = pow((Ix + Iy) , 0.5)
    
    theta = np.arctan(Iy / (Ix + 1e-8)) * (180 / np.pi)
    theta[Ix < 0] += 180
    theta = (theta + 360) % 360
    
    bit_size = 360 / bit
    theta_bits = (theta + (bit_size / 2)) // int(bit_size) % bit
    
    orient = np.zeros((bit,) + Ix.shape)
    for i in range(bit):
        orient[i][theta_bits == i] = 1
        orient[i] = Igm * orient[i]
        orient[i] = cv2.GaussianBlur(orient[i], blursize , 0)

    for s in range( len(fdt_image_value[0]) ):
        y = fdt_image_value[0][s]
        x = fdt_image_value[1][s]
        Vec = vector12( y , x , theta , orient)
        Vec = np.array(Vec)
        #still remain problem
        if np.sum(Vec) > 0:
            fd.append( [y , x , Vec] )

    print( '\n')
    print('Feature Description Complete')

    return fd



def main():
    image = cv2.imread(infile)
    feature_points, Ix , Iy = feature_detect(image)
    pointimage = pointview( infile , feature_points)
    cv2.imwrite( outfile , pointimage)
    fd = feature_descripter( image , feature_points, Ix , Iy)



if __name__=='__main__':
    main()

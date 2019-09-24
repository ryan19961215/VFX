#!/usr/bin/python
# -*- coding: UTF-8 -*-
import cv2
import math
import numpy as np
import scipy
import matplotlib.pyplot as plt
import random

import feature as ft

# main可以做的事情：計算infile1和infile2的移動向量，並且輸出用來比較兩者之線圖

# 輸入圖片檔（main程式會改用myio.py讀檔）
infile1 = 'victoria/victoria1.jpg'
infile2 = 'victoria/victoria2.jpg'
# 輸出Harris Detect的圖名（主要為feature.py做）
outfile1 = 'new1.jpg'
outfile2 = 'new2.jpg'
# 輸出線的圖片
matching_image_name = 'matching.jpg'
ransac_image_name = 'ransacimg.jpg'

# 可調整變數
threshnumber = 1000 #最大總matching線數
threshratemax = 0.8 #最小容忍誤差（第一名跟第二名的倍率，太接近怕誤判，避免類似風景）
ransac_time = 2000 #ransac的次數，愈大會有愈多的信心結果是好的
ransac_num = 8 #一次ransac取的隨機點數，一樣愈大通常愈好
var_ratio = 1.0 #若結果在一個變異數內給過，愈大的話結果會愈多，但是愈雜亂


def takemin(elem):
    return elem[4]


def feature_sort( fea_image , ori_image):
    #可做可不做，目的在於能夠讓程式自動化知道哪圖片的順序，最萬全的方法就是比較他們對到的點數
    #另外一種比較不安全抓順序的方法是抓圖片的建立時間
    return ori_image


def lineview(fea_record, img1 , img2 , string):
    h1 , w1 , _  = img1.shape
    h2 , w2 , _  = img2.shape
    vis = np.zeros([max(h1, h2), w1 + w2, 3]) + 255
    vis[:h1, :w1] = img1
    vis[:h2, w1:] = img2
    

    for element in fea_record:
        #print(element)#can print in csv file maybe later
        y1, x1 = element[0], element[1]
        y2, x2 = element[2], element[3]
        r1 = random.random()*255
        r2 = random.random()*255
        r3 = random.random()*255
        cv2.line( vis , (x1,y1) , (w1+x2,y2) , (r1,r2,r3) , 1 )

    cv2.imwrite(string , vis)

    return



def pair_matching(fea1_img , fea2_img):
    fea_info = []
    for fea_point1 in fea1_img:
        min = math.inf
        for fea_point2 in fea2_img:
            dis = np.linalg.norm(fea_point2[2] - fea_point1[2])
            if dis < min:
                rate = dis / min
                min = dis
                record = fea_point2[0] , fea_point2[1]
        if rate < threshratemax:
            fea_info.append( [fea_point1[0] , fea_point1[1] , record[0], record[1] , min , rate ])
        fea_info.sort(key=takemin,reverse=False)
        fea_chosen = fea_info[:threshnumber]

    return fea_chosen



def pair_check(pos,neg):
    match = []
    for pair1 in pos:
        for pair2 in neg:
            if pair1[0] == pair2[2] and pair1[1] == pair2[3] and pair1[2] == pair2[0] and pair1[3] == pair2[1]:
                match.append( pair1 )
                break
    print( '\n')
    print( 'There are' ,len(match), 'line(s) before ransac.' )

    return match



def feature_matching(fd_set , ori_image):
    image_num = 0
    match_image = []
    for fea1_img in fd_set:
        image_num = image_num + 1
        if image_num == len(fd_set):
            break
        fea2_img = fd_set[image_num]
        #比較：fea1_img和fea2_img，順向做一次反向也做一次
        pos_match = pair_matching(fea1_img , fea2_img)
        neg_match = pair_matching(fea2_img , fea1_img)
        match = pair_check(pos_match,neg_match)
        lineview(match, ori_image[image_num-1] , ori_image[image_num],matching_image_name)
        match_image.append( match )


    print('Feature Matching Complete')

    return match_image



def ransac( fd_set , match_image , ori_image ):
    move_image = []
    #match_image = np.array( match_image)
    for image in match_image:
        err = []
        dif = []
        match_num = len(image[0])
        image = np.array(image)
        for i in range(ransac_time):
            sample = [random.randint(0,match_num-1) for j in range(ransac_num)]
            sample_x = []
            sample_y = []
            for example in sample:
                sample_x.append(image[example][1] - image[example][3])
                sample_y.append(image[example][0] - image[example][2])
            mean_x = np.mean(sample_x)
            mean_y = np.mean(sample_y)
            dif_x = 0
            dif_y = 0
            for i,element in enumerate(sample):
                dif_x += np.abs(sample_x[i] - mean_x)
                dif_y += np.abs(sample_y[i] - mean_y)
            err.append( dif_x+dif_y )
            dif.append( [mean_y,mean_x] )
        err = np.array(err)
        ers = err.argsort()
        besty , bestx = dif[ers[0]]
        move_image.append(dif[ers[0]])

    new_match_image = []
    count = 0
    for image in match_image:
        setx = []
        sety = []
        for point in image:
            setx.append((point[1] - point[3]) - bestx)
            sety.append((point[0] - point[2]) - besty)
        xmean = np.mean( setx )
        ymean = np.mean( sety )
        xvar = np.var( setx )
        yvar = np.var( sety )
        for feature in image:
            xdif = np.abs(feature[1] - feature[3]  - bestx - xvar)
            ydif = np.abs(feature[0] - feature[2]  - besty - yvar)
            if xdif < (xvar * var_ratio) and ydif < (yvar * var_ratio):
                new_match_image.append( feature )

        lineview(new_match_image, ori_image[count] , ori_image[count+1] , ransac_image_name)
        count += 1
        print( '\n')
        print( 'There are', len(new_match_image) ,'line(s) after ransac.' )
        
    print('Image Matching Complete')

    print(move_image)

    return move_image



def main():
    image1 = cv2.imread(infile1)
    image2 = cv2.imread(infile2)
    feature_points1 , Ix1 , Iy1 = ft.feature_detect(image1)
    feature_points2 , Ix2 , Iy2 = ft.feature_detect(image2)
    pointimage1 = ft.pointview( infile1 , feature_points1)
    pointimage2 = ft.pointview( infile2 , feature_points2)
    cv2.imwrite( outfile1 , pointimage1)
    cv2.imwrite( outfile2 , pointimage2)
    fd1 = ft.feature_descripter( image1 , feature_points1 , Ix1 , Iy1)
    fd2 = ft.feature_descripter( image2 , feature_points2 , Ix2 , Iy2)
    image = []
    fd_set = []
    image.append(image1)
    image.append(image2)
    fd_set.append(fd1)
    fd_set.append(fd2)
    sort_image = feature_sort(fd_set,image)
    match_image = feature_matching(fd_set ,sort_image)
    move_image = ransac( fd_set , match_image , image)



if __name__=='__main__':
    main()

#!/usr/bin/python
# -*- coding: UTF-8 -*-
#這個程式可以刪掉沒關係
#這個程式可以刪掉沒關係
#這個程式可以刪掉沒關係
#用來存一些可能會用到的函式而已
#用來存一些可能會用到的函式而已
#用來存一些可能會用到的函式而已

import cv2
import math
import numpy as np
import scipy
import matplotlib.pyplot as plt

window = 2

def feature_descripter(colorimage , fdt_image_value , Ix , Iy):
    fd = []
    for s in range( len(fdt_image_value[0]) ):
        y = fdt_image_value[1][s]
        x = fdt_image_value[0][s]
        Vec = colorimage[y-window:y+window+1,x-window:x+window+1]
        return Vec


def feature_matching_(fd_set , ori_image):
    image_num = 0
    match_image = []
    for fea1_img in fd_set:
        image_num = image_num + 1
        if image_num == len(fd_set):
            break
        fea2_img = fd_set[image_num]
        fea_info = []
        fea_min = []
        #比較：fea1_img和fea2_img（兩個都是以一張圖為單位）
        for fea_point1 in fea1_img:
            min = math.inf
            for fea_point2 in fea2_img:
                dis = np.linalg.norm(fea_point2[2] - fea_point1[2])
                if dis < min:
                    rate = dis / min
                    min = dis
                    record = fea_point2[0] , fea_point2[1]
                if rate < threshratemax:#?
                    min = math.inf
        fea_info.append( [fea_point1[0] , fea_point1[1] , record[0], record[1] , min , rate ])
        fea_info.sort(key=takemin,reverse=False)
        record_num = 0
        fea_point = []
        for fea_record in fea_info:
            if record_num >= threshnumber:
                break
            if fea_record[5] > threshratemax:
                continue
            fea_point.append(fea_record[0:4])
            record_num = record_num + 1
    lineview(fea_point, ori_image[image_num-1] , ori_image[image_num] , record_num)
    match_image.append( fea_point )

    return match_image

def main():
    feature_descripter(colorimage , fdt_image_value , Ix , Iy)

if __name__=='__main__':
    main()

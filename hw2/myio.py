#!/usr/bin/python
# -*- coding: UTF-8 -*-
import cv2
import math
import numpy as np
import scipy
import matplotlib.pyplot as plt
import os


def imgread(sourcefilename):
    array_of_img = []
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    for filename in os.listdir(r"./"+sourcefilename):
        #print(filename) #just for test
        #img is used to store the image data
        img = cv2.imread(sourcefilename + "/" + filename)
        array_of_img.append(img)
        #print(img)
    print( "There are {:d} picture in the file.".format(len(array_of_img)))
    print( "I/O complete" )
    return array_of_img


def imgwrite(outputname , image):
    cv2.imwrite( outputname , image )
    return


def main():
    filename = 'cabin'
    outputname = 'wayne.jpg'
    array_of_img = imgread(filename)
    imgwrite(outputname,array_of_img[0])


if __name__=='__main__':
    main()



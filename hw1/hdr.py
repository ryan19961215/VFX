import PIL.Image
import PIL.ExifTags

import os
import sys

import cv2
import numpy as np

import array
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from tone_map import tone_map
import img_align
from img_align import img_align

'''
    # running this program by the following command: python3 hdr.py <directory of images> <number of points>
    # running under python 3.6.5 with Pillow
'''

class File():
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
    
    def __enter__(self):
        print("file open：" + self.filename)
        self.open_file = open(self.filename, self.mode)
        return self.open_file
    
    def __exit__(self, type, value, traceback):
        print("file close：" + self.filename)
        self.open_file.close()

def OptimizeEquation(imgs, exptime, points):
    nPts = len(points)
    nImgs = imgs.shape[0]
    #median
    w = np.arange(256)
    w[w>127] = 255.-w[w>127]
    A = np.zeros((nImgs * nPts + 255, 256+nPts), np.float32)
    b = np.zeros((nImgs * nPts + 255, 1), np.float32)
    k = 0

    #A and b
    for j, img in enumerate(imgs):
        for i, point in enumerate(points):
            z = img[point]
            A[k, z] = w[z]
            A[k, i+256] = -w[z]
            b[k] = exptime[j] * w[z]
            k += 1
    k += 1
    A[k, 127] = 1.

    for i in range(254):
        A[k, i] = w[i+1]
        A[k, i+1] = -2.*w[i+1]
        A[k, i+2] = w[i+1]
        k+=1

    #response
    invA = np.linalg.pinv(A)
    x = np.dot(invA, b)
    g = x[:256].reshape(-1)
    le = x[256:].reshape(-1)

    return g, le




# input
if len(sys.argv) != 3:
    print( 'Format wrong. Please run this program in the following format' )
    print( 'python3 hdr.py <directory of images> points' )
    sys.exit(-1)

else:
    imglist = os.listdir( sys.argv[1] )
print( imglist)

# image construction and exposure time
imgcont = []
exptime = []
for img in imglist:

    imgr = PIL.Image.open(os.path.abspath( os.path.join( sys.argv[1] , img ) ) )

    fn = cv2.imread( os.path.abspath( os.path.join( sys.argv[1] , img ) ) )

    imgcont.append( fn )
    if ( 1 ):
        exif = {
        PIL.ExifTags.TAGS[k]: v
        for k, v in imgr._getexif().items()
        if k in PIL.ExifTags.TAGS
    }  # solution by https://stackoverflow.com/questions/4764932/in-python-how-do-i-read-the-exif-data-for-an-image
        exposure_unit = float(float(exif['ExposureTime'][0]) / float(exif['ExposureTime'][1]) )
    else:
        print( 'please enter the expo time of ' + str(img) )
        exposure_unit = input()
    exptime.append( float(exposure_unit) )
    print('picture'+str(len( exptime)))

imgcont = np.array(imgcont)
lexpotime = np.log(exptime)

#到這裡 下面改成我自己的align
# img alignment
# alighMTB = cv2.createAlignMTB()
# alighMTB.process( imgcont, imgcont)
imgcont = img_align(imgcont)

# random checkpoint
Image_number, Image_height, Image_width, Image_color = imgcont.shape
pts = [(np.random.randint(0, Image_height), np.random.randint(0, Image_width)) for _ in range(int(sys.argv[2]))]

#optimize function
gr, ler = OptimizeEquation(imgcont[:, :, :, 2].reshape(-1, Image_height, Image_width), lexpotime, pts)
gg, leg = OptimizeEquation(imgcont[:, :, :, 1].reshape(-1, Image_height, Image_width), lexpotime, pts)
gb, leb = OptimizeEquation(imgcont[:, :, :, 0].reshape(-1, Image_height, Image_width), lexpotime, pts)

# Generate the radiance map
radm = np.zeros((Image_height, Image_width, 3))
eps = 1e-10
w = np.arange(256)
w[w>127] = 255.-w[w>127]

for r in range(Image_height):
    for c in range(Image_width):
        zr = imgcont[:, r, c, 2]
        zg = imgcont[:, r, c, 1]
        zb = imgcont[:, r, c, 0]
        radm[r, c, 2] = np.exp(np.sum(w[zr]*(gr[zr]-lexpotime))/(np.sum(w[zr])+eps))
        radm[r, c, 1] = np.exp(np.sum(w[zg]*(gg[zg]-lexpotime))/(np.sum(w[zg])+eps))
        radm[r, c, 0] = np.exp(np.sum(w[zb]*(gb[zb]-lexpotime))/(np.sum(w[zb])+eps))


# Output the hdr image
#radm = np.uint8(radm)
#radm2 = cv2.cvtColor(radm, cv2.COLOR_RGB2BGR)
# hdr_image = np.zeros(imgcont[0].shape, dtype=np.float64)
# hdr_image[..., channel] = cv2.normalize(radm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

cv2.imwrite('output.hdr', radm)

# print("Tonemaping using Durand's method ... ")
# tonemapDurand = cv2.createTonemapDurand(1.5,4,1.0,1,1)
# ldrDurand = tonemapDurand.process(radm)
# ldrDurand = 3 * ldrDurand
# cv2.imwrite("ldr-Durand.jpg", ldrDurand * 255)
# print("saved ldr-Durand.jpg")

tone_map(radm)

# Output radiance map
rad = np.log(radm)

rad[:, :, 0] = (rad[:, :, 0]-rad[:, :, 0].min())/(rad[:, :, 0].max()-rad[:, :, 0].min()+eps)*255
rad[:, :, 1] = (rad[:, :, 1]-rad[:, :, 1].min())/(rad[:, :, 1].max()-rad[:, :, 1].min()+eps)*255
rad[:, :, 2] = (rad[:, :, 2]-rad[:, :, 2].min())/(rad[:, :, 2].max()-rad[:, :, 2].min()+eps)*255

rad = cv2.applyColorMap(rad.astype(np.uint8), cv2.COLORMAP_JET)
cv2.imwrite('rad.png', rad)

# Show the draw points
for pt in pts:
    cv2.circle(imgcont[0], pt[::-1], 3, (0, 0, 255))

cv2.imwrite('pts.jpg', imgcont[0])









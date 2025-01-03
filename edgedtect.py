import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
os.chdir('C:/Users/acer/Desktop/DS & ML/tutorial/VIPER labs MMU/[2430] CDS6334 Assignment')
input_dir = 'dataset/test'
output_dir = 'dataset/output'
gt_dir = 'dataset/groundtruth'

# Get the list of image paths
bloodcells = glob.glob(os.path.join(input_dir, '*'))
# Read input grayscale image (fundus image or similar)
image = cv2.imread(bloodcells[0], cv2.IMREAD_GRAYSCALE)
sobelx = cv2.Sobel(image,cv2.CV_8U,1,0,ksize=1)
sobely = cv2.Sobel(image,cv2.CV_8U,0,1,ksize=1)
#sobelxy = cv2.addWeighted
laplacian = cv2.Laplacian(image,cv2.CV_8U)


gauss = cv2.GaussianBlur(image, (5, 5), 0)
cannyg = cv2.Canny(gauss,25,35)
canny = cv2.Canny(image,25,35)
# Display
cv2.imshow('Original', image)
#cv2.imshow('sobelx',sobelx)
#cv2.imshow('sobely',sobely)
cv2.imshow('canny', canny)
cv2.imshow('cannyg', cannyg)
#cv2.imshow('laplacian', laplacian)

cv2.waitKey(0)
cv2.destroyAllWindows()
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
ret,th1 = cv2.threshold(image,100,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(th1,kernel,iterations = 1)
dilation = cv2.dilate(th1,kernel,iterations = 1)
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
  
# opening the image 
opening = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel1, iterations=1) 
# closing the image
closing = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel1, iterations=1) 
# print the output 
plt.subplot(131), plt.imshow(image, cmap='gray') 
plt.subplot(132), plt.imshow(opening, cmap='gray')
plt.subplot(133), plt.imshow(closing, cmap='gray')
plt.show()
plt.subplot(141), plt.imshow(image, cmap='gray')
plt.subplot(142), plt.imshow(th1, cmap='gray')
plt.subplot(143), plt.imshow(erosion, cmap='gray')
plt.subplot(144), plt.imshow(dilation, cmap='gray')
plt.show()


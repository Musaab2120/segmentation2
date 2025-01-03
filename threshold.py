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
ret,th1 = cv2.threshold(image,75,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,31,2)
th3 = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,2)
plt.figure(figsize=(12,6))
plt.subplot(141), plt.imshow(image,cmap='gray')
plt.title('Grayscale Image')
plt.xticks([]), plt.yticks([])
plt.subplot(142), plt.imshow(th1,cmap='gray')
plt.title('Binary Image (Threshold: '+str(ret)+')')
plt.xticks([]),plt.yticks([])
plt.subplot(143), plt.imshow(th2,cmap='gray')
plt.title('Adaptive Threshold(mean)'), plt.xticks([]), plt.yticks([])
plt.subplot(144), plt.imshow(th3,cmap='gray')
plt.title('Adaptive Threshold(gauss)'), plt.xticks([]), plt.yticks([])
plt.show()

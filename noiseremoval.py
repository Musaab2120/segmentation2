
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
imagehist = cv2.equalizeHist(image)
clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
clahe_image = clahe.apply(image)
blur1 = cv2.blur(image,(5,5))
gauss = cv2.GaussianBlur(image, (5, 5), 5)
blur3 = cv2.medianBlur(clahe_image,5)
bilateral = cv2.bilateralFilter(clahe_image, 15, 75, 75) 
# Display
'''plt.imshow(image,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.show()
plt.figure(figsize=(10,5))
plt.subplot(1,4,1), plt.imshow(gauss,cmap = 'gray')
plt.title('gauss'), plt.xticks([]), plt.yticks([])
plt.subplot(1,4,2), plt.imshow(blur1,cmap = 'gray')
plt.title('avg blurred'), plt.xticks([]), plt.yticks([])
plt.subplot(1,4,3), plt.imshow(blur3,cmap = 'gray')
plt.title('median blurred'), plt.xticks([]), plt.yticks([])
plt.subplot(1,4,4), plt.imshow(bilateral,cmap = 'gray')
plt.title('bilateral'), plt.xticks([]), plt.yticks([])
plt.show()'''
cv2.imshow('original',image)
cv2.imshow('clahe',clahe_image)
cv2.imshow('hist',imagehist)

#cv2.imshow('gauss',gauss)
#cv2.imshow('median',blur3)
cv2.waitKey(0)
cv2.destroyAllWindows()


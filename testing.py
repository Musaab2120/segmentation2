
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
blurr = cv2.GaussianBlur(image, (5, 5), 0)
# Apply CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_image = clahe.apply(image)
#ret,th1 = cv2.threshold(clahe_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
th2 = cv2.adaptiveThreshold(clahe_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,2)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))
opening = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel, iterations=1) 

cv2.imshow('CLAHE Enhanced', clahe_image)
cv2.imshow('th2',th2)

cv2.waitKey(0)
cv2.destroyAllWindows()
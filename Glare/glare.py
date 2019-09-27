# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 22:13:41 2018

@author: SaLeH
"""
import cv2 
import numpy as np

img=cv2.imread('2.jpg')
img_new1=cv2.resize(img, None, fx=10/img.shape[1], fy=10/img.shape[0], interpolation=cv2.INTER_AREA)
width=300
height=400
img_new=cv2.resize(img, None, fx=width/img.shape[1], fy=height/img.shape[0], interpolation=cv2.INTER_AREA)
glare=cv2.imshow('glare',img)

image=cv2.imread('1.jpg')
image=cv2.resize(image, None, fx=width/image.shape[1], fy=height/image.shape[0], interpolation=cv2.INTER_AREA)
cv2.imshow('face1',image)
overlay_img = np.ones(image.shape, np.uint8) * 255
overlay_img[270:int(270+10), 180:int(180+10)] = img_new1

# Create a mask and generate it's inverse.
gray_glasses = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)

#cv2.imshow("eynak1",gray_glasses)
ret, mask = cv2.threshold(gray_glasses, 251, 255, cv2.THRESH_BINARY)
#mask[np.where((mask == [0] ).all(axis = 1))] = [255]


mask_inv = cv2.bitwise_not(mask)

temp = cv2.bitwise_and(image, image, mask=mask)

temp2 = cv2.bitwise_and(overlay_img, overlay_img, mask=mask_inv)
cv2.imshow('temp',temp2)

final_img = cv2.add(temp, temp2)
#final_img = cv2.add(image, temp2)
    
imS = cv2.resize(final_img, (1366, 768))

cv2.imshow('Lets wear Glasses', final_img)
final_img=cv2.resize(final_img, None, fx=960/final_img.shape[1], fy=1280/final_img.shape[0], interpolation=cv2.INTER_AREA)
cv2.imwrite('withglare.jpg',final_img)
cv2.waitKey(0)

cv2.destroyAllWindows()
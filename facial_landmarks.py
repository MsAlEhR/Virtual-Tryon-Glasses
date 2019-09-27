# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 21:17:04 2017

@author: SaLeH
"""

from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2


def rotate(image,angle):
    
    h,w = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX,cY),-angle,1)        
   
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
     # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))





detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

image = cv2.imread("image3.jpg")
image = imutils.resize(image, width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 # detect faces in the grayscale image
rects = detector(gray, 1)

# loop over the face detections
for (i, rect) in enumerate(rects):
	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy
	# array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)
 
	# convert dlib's rectangle to a OpenCV-style bounding box
	# [i.e., (x, y, w, h)], then draw the face bounding box
	(x, y, w, h) = face_utils.rect_to_bb(rect)
	#cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
	# show the face number
    
	#cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
	#	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) ; 
             
 
	# loop over the (x, y)-coordinates for the facial landmarks
	# and draw them on the image
	for i in range(len(shape)):
		cv2.circle(image, (shape[i][0], shape[i][1]), 1, (0, 0, 255), -1)
		cv2.circle(image, (shape[27][0], shape[27][1]), 2, (255, 0, 0), -1)
		cv2.circle(image, (shape[0][0], shape[0][1]), 4, (255, 0, 0), -1)
        

glass_img = cv2.imread('gpdpzoom 54.png')
glass_img=rotate(glass_img,0)
#glasses_width =27 * abs(shape[16][1] - shape[0][1])
overlay_img = np.ones(image.shape, np.uint8) * 255
h_glasses, w_glasses = glass_img.shape[:2]
# scaling the glasses to size of k=lenght between to eyes
scaling_factor =1.45*abs(shape[16][0] - shape[0][0]) /w_glasses

cv2.imshow("overlaY1",glass_img)

overlay_glasses = cv2.resize(glass_img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
overlay_glasses[np.where((overlay_glasses == [0] ).all(axis = 0))] = [255] 
overlay_glasses[np.where((overlay_glasses == [0] ).all(axis = 1))] = [255]
overlay_glasses[np.where((overlay_glasses ==[0] ).all(axis = 2))] = [255]
cv2.imshow("overlaY",overlay_glasses)

edges = cv2.Canny(glass_img,50,90)
cv2.imshow("edge",edges)

# The x and y variables below depend upon the size of the detected face.
x = shape[0][0]-int((0.07)*overlay_glasses.shape[1])
y = shape[0][1]-int((0.37)*overlay_glasses.shape[1])

# Slice the height, width of the overlay image.
h, w = overlay_glasses.shape[:2]
overlay_img[int(y):int(y + h), int(x):int(x + w)] = overlay_glasses

# Create a mask and generate it's inverse.
gray_glasses = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)

#cv2.imshow("eynak1",gray_glasses)
ret, mask = cv2.threshold(gray_glasses, 130, 255, cv2.THRESH_BINARY)
#mask[np.where((mask == [0] ).all(axis = 1))] = [255]


mask_inv = cv2.bitwise_not(mask)

temp = cv2.bitwise_and(image, image, mask=mask)
   
temp2 = cv2.bitwise_and(overlay_img, overlay_img, mask=mask_inv)

#final_img = cv2.add(temp, temp2)
final_img = cv2.add(temp, temp2)
    
#imS = cv2.resize(final_img, (1366, 768))

cv2.imshow('Lets wear Glasses', final_img)
        
"""                  
	for (x, y) in shape:
		cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
"""
# show the output image with the face detections + facial landmarks
cv2.imshow("Output", image)

cv2.waitKey(0)

cv2.destroyAllWindows()



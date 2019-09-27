# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 20:22:21 2017

@author: mm
"""

# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2
import numpy as np

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

video =cv2.VideoWriter('face.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 8, (400,300))

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
vs = VideoStream().start()
#vs=cv2.VideoCapture("face.avi");
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream, resize it to
	# have a maximum width of 400 pixels, and convert it to
	# grayscale
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
	# detect faces in the grayscale frame
	rects = detector(gray, 0)
 
     	# loop over the face detections
 
	for rect in rects:
     
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
      
		shape = face_utils.shape_to_np(shape)
            
		glass_img = cv2.imread('front_image.png')
        #glasses_width =27 * abs(shape[16][1] - shape[0][1])
		overlay_img = np.ones(frame.shape, np.uint8) * 255
		h_glasses, w_glasses = glass_img.shape[:2]
        # scaling the glasses to size of k=lenght between to eyes
		scaling_factor =1.25*abs(shape[16][0] - shape[0][0]) /w_glasses
        

        
		overlay_glasses = cv2.resize(glass_img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
		overlay_glasses[np.where((overlay_glasses == [0] ).all(axis = 0))] = [255] 
		overlay_glasses[np.where((overlay_glasses == [0] ).all(axis = 1))] = [255]
		overlay_glasses[np.where((overlay_glasses ==[0] ).all(axis = 2))] = [255]

        # The x and y variables below depend upon the size of the detected face.
		x = shape[0][0]-int((0.07)*overlay_glasses.shape[1])
		y = shape[0][1]-int((0.22)*overlay_glasses.shape[1])
        
        # Slice the height, width of the overlay image.
		h, w = overlay_glasses.shape[:2]
		overlay_img[int(y):int(y + h), int(x):int(x + w)] = overlay_glasses
        
        # Create a mask and generate it's inverse.
		gray_glasses = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)
        
        #cv2.imshow("eynak1",gray_glasses)
		ret, mask = cv2.threshold(gray_glasses, 90, 255, cv2.THRESH_BINARY)
        #mask[np.where((mask == [0] ).all(axis = 1))] = [255]
        
        
		mask_inv = cv2.bitwise_not(mask)
        
		temp = cv2.bitwise_and(frame, frame, mask=mask)
           
		temp2 = cv2.bitwise_and(overlay_img, overlay_img, mask=mask_inv)
        
        
		final_img = cv2.add(temp, temp2)
        
	# show the frame
	cv2.imshow("Frame", final_img)
	video.write(final_img)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break 



cv2.destroyAllWindows()
vs.stop()
import numpy as np
import cv2
import dlib 
import imutils
from imutils import face_utils

cap =  cv2.VideoCapture("outpy.avi")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
num_frame=0

while(cap.isOpened()):
    ret, frame = cap.read()
    
    frame = imutils.resize(frame, width=400)
    num_frame=num_frame+1
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
	    	cv2.imshow('glass_img',glass_img)
          #glasses_width =27 * abs(shape[16][1] - shape[0][1])
	    	overlay_img = np.ones(frame.shape, np.uint8) * 255
	    	h_glasses, w_glasses = glass_img.shape[:2]
           # scaling the glasses to size of k=lenght between to eyes
	    	scaling_factor =1.25*abs(shape[16][0] - shape[0][0]) /w_glasses



	    	overlay_glasses = cv2.resize(glass_img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
	    	#overlay_glasses[np.where((overlay_glasses == [0] ).all(axis = 0))] = [255] 
	    	#overlay_glasses[np.where((overlay_glasses == [0] ).all(axis = 1))] = [255]
	    	#overlay_glasses[np.where((overlay_glasses ==[0] ).all(axis = 2))] = [255]
	    	cv2.imshow('glass_img',overlay_glasses) 
           # The x and y variables below depend upon the size of the detected face.
	    	x = shape[0][0]-int((0.2)*overlay_glasses.shape[0])
	    	y = shape[0][1]-int((0.3)*overlay_glasses.shape[1])

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
	    	cv2.imshow('temp',temp2)  

	    	final_img = cv2.add(temp, temp2)

  
    cv2.imshow('frame',final_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        		       break


cap.release()
cv2.destroyAllWindows()


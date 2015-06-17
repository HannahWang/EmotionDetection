import cv
import cv2
import sys
import os
import math
import numpy as np
from matplotlib import pyplot as plt
#import matlab.engine

#cascPath = sys.argv[1]
cascPath_face = 'data/haarcascades/haarcascade_frontalface_default.xml'
cascPath_eye = 'data/haarcascades/haarcascade_eye.xml'
cascPath_pupil = 'data/haarcascades/haarcascade_eye_tree_eyeglasses.xml'
cascPath_mouth = 'data/haarcascades/haarcascade_mcs_mouth.xml'
cascPath_smile = 'data/haarcascades/haarcascade_smile.xml'
faceCascade = cv2.CascadeClassifier(cascPath_face)
eyeCascade = cv2.CascadeClassifier(cascPath_eye)
pupilCascade = cv2.CascadeClassifier(cascPath_pupil)
mouthCascade = cv2.CascadeClassifier(cascPath_mouth)
smileCascade = cv2.CascadeClassifier(cascPath_smile)

video_capture = cv2.VideoCapture(0)

Happy = False
SAD = False
NOEMOTION = False
font = cv2.FONT_HERSHEY_SIMPLEX
name = ''
#name = input('Do you have emotion?')
#if name.isspace():
#	NOEMOTION = True
while True:
	# Capture frame-by-frame
	ret, frame = video_capture.read()
	
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=2,
		minNeighbors=5,
		minSize=(30, 30),
		flags=cv2.cv.CV_HAAR_SCALE_IMAGE
	)
	# Draw a rectangle around the faces
	j = 0
	face = []
	for (x, y, w, h) in faces:
		fx = int(x + 0.1*w)
		fy = y
		fw = int(0.85*w)
		fh = h
		face.append([fx,fy,fw,fh])
		cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 0, 0), 2)
		face_gray = gray[fy:fy+fh, fx:fx+fw]
		face_color = frame[fy:fy+fh, fx:fx+fw]
		

		# Draw circles at the pupils & draw rectangles around the eyes
		pupils = pupilCascade.detectMultiScale(
			face_gray,
			scaleFactor = 2,
			minNeighbors = 5,
			minSize = (10,10),
		)
		print "pupils:"
		print pupils
		i = 0
		d = [[0,0], [0,0]]
		if len(pupils) == 2:
			for (px,py,pw,ph) in pupils:
				center_px = int(px+pw*0.5)
				center_py = int(py+pw*0.5)
				radius = 5
				cv2.circle(face_color,(center_px,center_py), radius, (0,0,255),-1)
				d[i][0] = center_px
				d[i][1] = center_py
				i = i + 1
			distance = math.sqrt((d[1][0]-d[0][0])**2 + (d[1][1]-d[1][0])**2)
			eyes = []
			for i in range(0,2):
				left = int(d[i][0] - distance/5)
				top = int(d[i][1] - distance/8.7)
				right = int(d[i][0] + distance/5)
				bottom = int(d[i][1] +distance/8.7)
				eyes.append([left, top, int(distance*2/3), int(distance/3)])
				cv2.rectangle(face_color,(left,top),(right,bottom),(0,255,0),2)
			if eyes[0][0] > eyes[1][0]:
				tmp = eyes[1]
				eyes[1] = eyes[0]
				eyes[0] = tmp
			print "eyes:"
			print eyes
			
		eyebrows = eyeCascade.detectMultiScale(
			face_gray,
			scaleFactor = 2,
			minNeighbors = 5,
			minSize = (10,10),
		)
		#eye_color = []
		#print "eyes:"
		#print eyes
		if len(eyebrows) == 2:
			eyebrows = []
			for (ex,ey,ew,eh) in eyebrows:
				cv2.rectangle(face_color,(ex,ey),(ex+ew,ey+eh), (0,255,0),2)
				eyebrows.append([ex-15, ey-30, ex+ew+15, ey+eh-55])
				#eye_color.append(face_color[ey-30:ey+eh-55,ex-15:ex+ew+15])
			#if eyebrows[0][0] > eyebrows[1][0]:
			#	tmp = eyebrows[1]
			#	eyebrows[1] = eyebrows[0]
			#	eyebrows[0] = tmp
		#sobelx64f1 = cv2.Sobel(eye_color[0],cv2.CV_64F,0,1,ksize=5)
		#abs_sobel64f1 = np.absolute(sobelx64f1)
		#sobel_8u1 = np.uint8(abs_sobel64f1)
		#sobelx64f2 = cv2.Sobel(eye_color[1],cv2.CV_64F,0,1,ksize=5)
		#abs_sobel64f2 = np.absolute(sobelx64f2)
		#sobel_8u2 = np.uint8(abs_sobel64f2)
		#plt.subplot(2,2,1),plt.imshow(eye_color[0],cmap = 'gray')
		#plt.title('Original'), plt.xticks([]), plt.yticks([])
		#plt.subplot(2,2,2),plt.imshow(sobel_8u1,cmap = 'gray')
		#plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])
		#plt.subplot(2,2,3),plt.imshow(eye_color[1],cmap = 'gray')
		#plt.title('Original'), plt.xticks([]), plt.yticks([])
		#plt.subplot(2,2,4),plt.imshow(sobel_8u2,cmap = 'gray')
		#plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])
		#plt.show()
		
		#mouth = []
		#mx = int(fx + fw/3)
		#my = int(fy + fh*7/10)
		#mw = int(fw/2)
		#mh = int(fh/5)
		#mouth.append([mx,my,mw,mh])
		#cv2.rectangle(face_color, (mx,my), (mx+mw,my+mh), (255,255,255), 2)
		#print "mouth:"
		#print mouth

	#	mouth = mouthCascade.detectMultiScale(
	#		face_gray,
	#		scaleFactor = 6.5,
	#		minNeighbors = 5,
	#		minSize = (10,10),
	#	)
	#	if len(mouth) == 1:
	#		print "mouth:" 
	#		print mouth
	#		for (mx, my, mw, mh) in mouth:
	#			cv2.rectangle(face_color,(mx,my),(mx+mw,my+mh), (255,255,255),2)
		
		smile = smileCascade.detectMultiScale(
			face_gray,
			scaleFactor = 6.5,
			minNeighbors = 5,
			minSize = (50,50),
		)
		#if not name.isspace():
		if len(smile) == 1:
			print "smile:" 
			print smile
			for (sx, sy, sw, sh) in smile:
				cv2.rectangle(face_color,(sx,sy),(sx+sw,sy+sh), (0,255,255),2)
			Happy = True
			SAD = False
		else:
			Happy = False
			SAD = True

	print "face:"
	print face
	print "\n"
	
	if Happy:
		cv2.putText(frame,'Happy',(100,100), font, 4,(50,50,255),2)
	if SAD:
		cv2.putText(frame,'Sad',(100,100), font, 4,(50,50,255),2)
	if NOEMOTION:
		cv2.putText(frame,'No emotion',(100,100), font, 4,(50,50,255),2)
	# Display the resulting frame
	cv2.imshow('Video', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

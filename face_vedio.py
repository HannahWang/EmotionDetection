import cv2
import sys
import os

#cascPath = sys.argv[1]
cascPath_face = 'data/haarcascades/haarcascade_frontalface_default.xml'
cascPath_eye = 'data/haarcascades/haarcascade_eye.xml'
cascPath_mouth = 'data/haarcascades/haarcascade_mcs_mouth.xml'
faceCascade = cv2.CascadeClassifier(cascPath_face)
eyeCascade = cv2.CascadeClassifier(cascPath_eye)
mouthCascade = cv2.CascadeClassifier(cascPath_mouth)

video_capture = cv2.VideoCapture(0)

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
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
		face_gray = gray[y:y+h, x:x+w]
		face_color = frame[y:y+h, x:x+w]
		eyes = eyeCascade.detectMultiScale(
			face_gray,
			scaleFactor = 2,
			minNeighbors = 5,
			minSize = (10,10),
		)
		print "eyes:\n"
		print eyes
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(face_color,(ex,ey),(ex+ew,ey+eh), (0,255,0),2)
		mouth = mouthCascade.detectMultiScale(
			face_gray,
			scaleFactor = 6.5,
			minNeighbors = 5,
			minSize = (10,10),
		)
		print "mouth:\n" 
		print mouth
		for (mx, my, mw, mh) in mouth:
			cv2.rectangle(face_color,(mx,my),(mx+mw,my+mh), (255,255,255),2)

	# Display the resulting frame
	cv2.imshow('Video', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

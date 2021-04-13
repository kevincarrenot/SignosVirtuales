import cv2

class VideoCamera(object):
	
	def __init__(self):
		self.video = cv2.VideoCapture(0)

	def __del__(self):
	 	self.video.release()

	def get_frame(self):
		#haar_file = 'haarcascade_frontalface_default.xml'
		#face_cascade = cv2.CascadeClassifier(haar_file)
		success, image = self.video.read()
		#face=face_cascade.detectMultiScale(image,1.1,7)
		#for (x,y,h,w) in face:
		#	cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
		ret, jpeg = cv2.imencode('.jpg', image)
		return jpeg.tobytes()
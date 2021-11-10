import numpy as np 
import cv2
import os
from tracker import Tracker
import time
import imageio

images = []
tracker = Tracker(50,3)
track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
				(127, 127, 255), (255, 0, 255), (255, 127, 255),
				(127, 0, 255), (127, 0, 127),(127, 10, 255), (0,255, 127)]


def task1_1(root = '../Sequences/01/'):	
	for img_name in os.listdir(root):
		img_path = os.path.join(root,img_name)
		img = cv2.imread(img_path)	
		frame = np.ones(img.shape,np.uint8)*255


		_, img_binarized = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY_INV)
		img_gray = cv2.cvtColor(img_binarized, cv2.COLOR_BGR2GRAY)
		img_gray = cv2.medianBlur(img_gray, 7) 

		contours, hierarchy = cv2.findContours(img_gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)	
		centers = []
		for c in contours:
			(x, y), radius = cv2.minEnclosingCircle(c)
			centers.append((x,y))

		centers = np.array(centers)

		if (len(centers) > 0):
			tracker.update(centers,contours)
			for j in tracker.cur_frame:
				x = int(tracker.tracks[j].trace[-1][0])
				y = int(tracker.tracks[j].trace[-1][1])
				track_color = track_colors[tracker.tracks[j].trackId % len(track_colors)]
				cv2.putText(frame,str(tracker.tracks[j].trackId), (x-10,y-20),0, 0.5, track_color,2)
				cv2.drawContours(frame,tracker.tracks[j].contour, -1, track_color, 3)


			cv2.imshow('image',frame)
			# cv2.imwrite("image"+str(i)+".jpg", frame)
			# images.append(imageio.imread("image"+str(i)+".jpg"))
			time.sleep(0.5)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				cv2.destroyAllWindows()
				break
			




def task1_2(root = '../Sequences/01/'):	
	for img_name in os.listdir(root):
		img_path = os.path.join(root,img_name)
		img = cv2.imread(img_path)	
		frame = np.ones(img.shape,np.uint8)*255


		_, img_binarized = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY_INV)
		img_gray = cv2.cvtColor(img_binarized, cv2.COLOR_BGR2GRAY)
		img_gray = cv2.medianBlur(img_gray, 7) 

		contours, hierarchy = cv2.findContours(img_gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)	
		centers = []
		for c in contours:
			(x, y), radius = cv2.minEnclosingCircle(c)
			centers.append((x,y))

		centers = np.array(centers)

		if (len(centers) > 0):
			tracker.update(centers,contours)
			for j in range(len(tracker.tracks)):
				if (len(tracker.tracks[j].trace) > 1):
					x = int(tracker.tracks[j].trace[-1][0])
					y = int(tracker.tracks[j].trace[-1][1])
					tl = (x-10,y-10)
					br = (x+10,y+10)
					track_color = track_colors[tracker.tracks[j].trackId % len(track_colors)]
					cv2.rectangle(frame,tl,br,track_color,1)
					cv2.putText(frame,str(tracker.tracks[j].trackId), (x-10,y-20),0, 0.5, track_color,2)
					for k in range(len(tracker.tracks[j].trace)):
						x = int(tracker.tracks[j].trace[k][0])
						y = int(tracker.tracks[j].trace[k][1])
						cv2.circle(frame,(x,y), 3, track_color,-1)
					cv2.circle(frame,(x,y), 6, track_color,-1)

			cv2.imshow('image',frame)
			# cv2.imwrite("image"+str(i)+".jpg", frame)
			# images.append(imageio.imread("image"+str(i)+".jpg"))
			time.sleep(0.5)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				cv2.destroyAllWindows()
				break

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          

if __name__ == '__main__':
	task1_1('../Sequences/01/')
	#task1_2()
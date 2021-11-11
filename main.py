import numpy as np 
import cv2
import os
from tracker import Tracker
import time
import imageio

images = []
tracker = Tracker(40,3,200)
track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
				(127, 127, 255), (255, 0, 255), (255, 127, 255),
				(127, 0, 255), (127, 0, 127),(127, 10, 255), (0,255, 127)]


# def task1_1(root = '../Sequences/01/'):	
# 	i = 0
# 	for img_name in os.listdir(root):
# 		img_path = os.path.join(root,img_name)
# 		img = cv2.imread(img_path)	
# 		frame = np.ones(img.shape,np.uint8)*255


# 		_, img_binarized = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY_INV)
# 		img_gray = cv2.cvtColor(img_binarized, cv2.COLOR_BGR2GRAY)
# 		img_gray = cv2.medianBlur(img_gray, 7) 

# 		_, contours, hierarchy = cv2.findContours(img_gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)	
# 		centers = []
# 		for c in contours:
# 			(x, y), radius = cv2.minEnclosingCircle(c)
# 			centers.append((x,y))

# 		centers = np.array(centers)

# 		if (len(centers) > 0):
# 			tracker.update(centers,contours)
# 			for j in tracker.cur_frame:
# 				x = int(tracker.tracks[j].trace[-1][0])
# 				y = int(tracker.tracks[j].trace[-1][1])
# 				track_color = track_colors[tracker.tracks[j].trackId % len(track_colors)]
# 				cv2.putText(frame,str(tracker.tracks[j].trackId), (x-10,y-20),0, 0.5, track_color,2)
# 				cv2.drawContours(frame,tracker.tracks[j].contour, -1, track_color, 3)


# 			cv2.imshow('image',frame)
# 			# cv2.imwrite("image"+str(i)+".jpg", frame)
# 			# images.append(imageio.imread("image"+str(i)+".jpg"))
# 			i += 1
# 			time.sleep(0.5)
# 			if cv2.waitKey(1) & 0xFF == ord('q'):
# 				cv2.destroyAllWindows()
# 				break
			




def main(root = '../Sequences/01/'):	
	initial = True
	coords = {}
	for img_name in os.listdir(root):
		img_path = os.path.join(root,img_name)
		img = cv2.imread(img_path)	
		frame = np.ones(img.shape,np.uint8)*255


		_, img_binarized = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY_INV)
		img_gray = cv2.cvtColor(img_binarized, cv2.COLOR_BGR2GRAY)
		img_gray = cv2.medianBlur(img_gray, 7) 

		_, contours, hierarchy = cv2.findContours(img_gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)	
		centers = []
		for c in contours:
			(x, y), radius = cv2.minEnclosingCircle(c)
			centers.append((x,y))

		centers = np.array(centers)
		print(centers)

		if (len(centers) > 0):
			# 1.2
			print(len(centers))
			tracker.update(centers, contours)
			# for j in range(len(tracker.tracks)):
			# 	print(str(tracker.tracks[j].trackId) + str(tracker.tracks[j].trace))
			distance = 0
			active_trackers = 0
			coords_new = {}
			for j in range(len(tracker.tracks)):
				if (len(tracker.tracks[j].trace) > 1):
					coords_new[tracker.tracks[j].trackId] = tracker.tracks[j].trace[-1]
					x = int(tracker.tracks[j].trace[-1][0,0])
					y = int(tracker.tracks[j].trace[-1][0,1])
					tl = (x-10,y-10)
					br = (x+10,y+10)
					track_color = track_colors[tracker.tracks[j].trackId % len(track_colors)]
					cv2.rectangle(frame,tl,br,track_color,1)
					cv2.putText(frame,str(tracker.tracks[j].trackId), (x-10,y-20),0, 0.5, track_color,2)
					cv2.drawContours(frame,tracker.tracks[j].contour, -1, track_color, 3)
					for k in range(len(tracker.tracks[j].trace)):
						x = int(tracker.tracks[j].trace[k][0,0])
						y = int(tracker.tracks[j].trace[k][0,1])
						cv2.circle(frame,(x,y), 3, track_color,-1)
					cv2.circle(frame,(x,y), 6, track_color,-1)

			# 2.3
			for j in range(len(tracker.tracks)):
				if (len(tracker.tracks[j].trace) > 1):
					# print(tracker.tracks[j].trace[-1])
					# print(coords_new)
					# print(coords)
					try:
						d = np.sqrt((coords_new[tracker.tracks[j].trackId][0,0]-coords[tracker.tracks[j].trackId][0,0])**2+(coords_new[tracker.tracks[j].trackId][0,1]-coords[tracker.tracks[j].trackId][0,1])**2)
						if d > 0:
							distance += d
							active_trackers += 1
							print(str(tracker.tracks[j].trackId) + ": " + str(distance) + " active trackers " + str(active_trackers))
						# print(coords[tracker.tracks[j].trackId])
						coords[tracker.tracks[j].trackId] = coords_new[tracker.tracks[j].trackId]
					except KeyError:
						coords[tracker.tracks[j].trackId] = coords_new[tracker.tracks[j].trackId]
						
			if active_trackers == 0:
				print(img_name + " Need second image to calculate avg displacement")
			else:
				distance = distance / active_trackers
				print(img_name + " Avg displacement: " + str(distance))

			cv2.imshow('image',frame)
			# cv2.imwrite("image"+str(i)+".jpg", frame)
			# images.append(imageio.imread("image"+str(i)+".jpg"))
			time.sleep(0.5)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				cv2.destroyAllWindows()
				break

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          

if __name__ == '__main__':
	# task1_1()
	main()
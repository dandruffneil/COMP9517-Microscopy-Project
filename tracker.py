import numpy as np 
from kalmanFilter import KalmanFilter
from scipy.optimize import linear_sum_assignment
from collections import deque

# class Tracks(object):
# 	def __init__(self, detection, trackId):
# 		super(Tracks, self).__init__()
# 		self.KF = KalmanFilter()
# 		self.KF.predict()
# 		self.KF.correct(np.matrix(detection).reshape(2,1))
# 		self.trace = deque(maxlen=200)
# 		self.prediction = detection.reshape(1,2)
# 		self.trackId = trackId
# 		self.skipped_frames = 0

# 	def predict(self,detection,contour):
# 		# pred = self.KF.predict()
# 		# if len(self.trace)==0:
# 		# 	self.prediction = detection.reshape(1,2)
# 		# else:
# 		# 	self.prediction = np.array(pred).reshape(1,2)
# 		# self.KF.correct(np.matrix(detection).reshape(2,1))
# 		# self.trace.append(detection)
# 		self.contour = contour
# 		self.prediction = np.array(self.KF.predict()).reshape(1,2)
# 		self.KF.correct(np.matrix(detection).reshape(2,1))
	
# 	# def self_play(self):
# 	# 	self.trace.append(self.KF.predict())


# class Tracker(object):
# 	def __init__(self, dist_threshold, max_frame_lost):
# 		super(Tracker, self).__init__()
# 		self.dist_threshold = dist_threshold
# 		self.max_frame_lost = max_frame_lost
# 		self.trackId = 0
# 		self.tracks = []
# 		# self.cur_frame = []
# 		# self.last_frame = []


# 	def update(self, detections,contours):
# 		# self.cur_frame = []
# 		if len(self.tracks) == 0:
# 			for i in range(detections.shape[0]):
# 				track = Tracks(detections[i],self.trackId)
# 				# track.predict(detections[i],contours[i])
# 				self.trackId +=1
# 				self.tracks.append(track)
# 				# self.cur_frame.append(i)
# 			# self.last_frame = self.cur_frame
# 			# return

# 		N = len(self.tracks)
# 		M = len(detections)
# 		cost = []
# 		for i in range(N):
# 			# ind = self.last_frame[i]
# 			print(str(self.tracks[i].trackId) + str(self.tracks[i].trace))
# 			diff = np.linalg.norm(self.tracks[i].prediction - detections.reshape(-1,2), axis=1)
# 			cost.append(diff)

# 		cost = np.array(cost)*0.1
# 		row, col = linear_sum_assignment(cost)
# 		assignment = [-1]*N
# 		for i in range(len(row)):
# 			assignment[row[i]] = col[i]

# 		unassigned_tracks = []

# 		for i in range(len(assignment)):
# 			if assignment[i] != -1:
# 				if (cost[i][assignment[i]] > self.dist_threshold):
# 					assignment[i] = -1
# 					unassigned_tracks.append(i)
# 					# self.tracks[i].lost_frames += 1
# 					#self.tracks[i].self_play
# 				else:
# 					# self.tracks[i].predict(detections[assignment[i]],contours[assignment[i]])
# 					# self.cur_frame.append(i)
# 					self.tracks[i].skipped_frames +=1
			
# 			# else:
# 			# 	unassigned_tracks.append(i)
# 			# 	self.tracks[i].lost_frames += 1
# 			# 	self.tracks[i].self_play
		



# 		del_tracks = []
# 		for i in range(len(self.tracks)):
# 			if self.tracks[i].skipped_frames > self.max_frame_lost:
# 				del_tracks.append(i)

# 		if len(del_tracks) > 0:
# 			for i in range(len(del_tracks)):
# 				del self.tracks[i]
# 				del assignment[i]

# 		for i in range(len(detections)):
# 			if i not in assignment:
# 				track = Tracks(detections[i],self.trackId)
# 				# track.predict(detections[i],contours[i])
# 				self.trackId +=1
# 				self.tracks.append(track)
# 				# self.cur_frame.append(len(self.tracks)-1)
		
# 		# self.last_frame = self.cur_frame
# 		for i in range(len(assignment)):
# 			if(assignment[i] != -1):
# 				self.tracks[i].skipped_frames = 0
# 				self.tracks[i].predict(detections[assignment[i]], contours[assignment[i]])
# 			self.tracks[i].trace.append(self.tracks[i].prediction)

import numpy as np 
from kalmanFilter import KalmanFilter
from scipy.optimize import linear_sum_assignment
from collections import deque


class Tracks(object):
	"""docstring for Tracks"""
	def __init__(self, detection, trackId):
		super(Tracks, self).__init__()
		self.KF = KalmanFilter()
		self.KF.predict()
		self.KF.correct(np.matrix(detection).reshape(2,1))
		self.trace = deque(maxlen=20)
		self.prediction = detection.reshape(1,2)
		self.trackId = trackId
		self.skipped_frames = 0

	def predict(self,detection,contour):
		self.prediction = np.array(self.KF.predict()).reshape(1,2)
		self.KF.correct(np.matrix(detection).reshape(2,1))
		self.contour = contour


class Tracker(object):
	"""docstring for Tracker"""
	def __init__(self, dist_threshold, max_frame_skipped, max_trace_length):
		super(Tracker, self).__init__()
		self.dist_threshold = dist_threshold
		self.max_frame_skipped = max_frame_skipped
		self.max_trace_length = max_trace_length
		self.trackId = 0
		self.tracks = []

	def update(self, detections,contours):
		if len(self.tracks) == 0:
			for i in range(detections.shape[0]):
				track = Tracks(detections[i], self.trackId)
				self.trackId +=1
				self.tracks.append(track)

		N = len(self.tracks)
		M = len(detections)
		cost = []
		for i in range(N):
			diff = np.linalg.norm(self.tracks[i].prediction - detections.reshape(-1,2), axis=1)
			cost.append(diff)

		cost = np.array(cost)*0.1
		row, col = linear_sum_assignment(cost)
		assignment = [-1]*N
		for i in range(len(row)):
			assignment[row[i]] = col[i]

		un_assigned_tracks = []

		for i in range(len(assignment)):
			if assignment[i] != -1:
				if (cost[i][assignment[i]] > self.dist_threshold):
					assignment[i] = -1
					un_assigned_tracks.append(i)
				else:
					self.tracks[i].skipped_frames +=1

		del_tracks = []
		for i in range(len(self.tracks)):
			if self.tracks[i].skipped_frames > self.max_frame_skipped :
				del_tracks.append(i)

		if len(del_tracks) > 0:
			for i in range(len(del_tracks)):
				del self.tracks[i]
				del assignment[i]

		for i in range(len(detections)):
			if i not in assignment:
				track = Tracks(detections[i], self.trackId)
				self.trackId +=1
				self.tracks.append(track)


		for i in range(len(assignment)):
			if(assignment[i] != -1):
				self.tracks[i].skipped_frames = 0
				self.tracks[i].predict(detections[assignment[i]], contours[assignment[i]])
			self.tracks[i].trace.append(self.tracks[i].prediction)

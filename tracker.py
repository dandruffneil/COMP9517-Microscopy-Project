import numpy as np 
from kalmanFilter import KalmanFilter
from scipy.optimize import linear_sum_assignment


class Tracks(object):
	def __init__(self, trackId, max_frame_lost):
		super(Tracks, self).__init__()
		self.KF = KalmanFilter()
		self.trace = []
		self.trackId = trackId
		self.prediction = None
		self.lost_frames = 0
		self.max_frame_lost = max_frame_lost
		self.contour = []

	def predict(self,detection,contour):
		pred = self.KF.predict()
		if len(self.trace)==0:
			self.prediction = detection.reshape(1,2)
		else:
			self.prediction = np.array(pred).reshape(1,2)
		self.KF.correct(np.matrix(detection).reshape(2,1))
		self.trace.append(detection)
		self.contour = contour
	
	def self_play(self):
		self.trace.append(self.KF.predict())


class Tracker(object):
	def __init__(self, dist_threshold, max_frame_lost):
		super(Tracker, self).__init__()
		self.dist_threshold = dist_threshold
		self.max_frame_lost = max_frame_lost
		self.trackId = 0
		self.tracks = []
		self.cur_frame = []
		self.last_frame = []


	def update(self, detections,contours):
		self.cur_frame = []
		if len(self.tracks) == 0:
			for i in range(detections.shape[0]):
				track = Tracks(self.trackId,self.max_frame_lost)
				track.predict(detections[i],contours[i])
				self.trackId +=1
				self.tracks.append(track)
				self.cur_frame.append(i)
			self.last_frame = self.cur_frame
			return

		N = len(self.last_frame)
		M = len(detections)
		cost = []
		for i in range(N):
			ind = self.last_frame[i]
			diff = np.linalg.norm(self.tracks[ind].prediction - detections.reshape(-1,2), axis=1)
			cost.append(diff)

		cost = np.array(cost)*0.1
		row, col = linear_sum_assignment(cost)
		assignment = [-1]*N
		for i in range(len(row)):
			assignment[row[i]] = col[i]

		unassigned_tracks = []

		for i in range(len(assignment)):
			if assignment[i] != -1:
				if (cost[i][assignment[i]] > self.dist_threshold):
					assignment[i] = -1
					unassigned_tracks.append(i)
					self.tracks[i].lost_frames += 1
					#self.tracks[i].self_play
				else:
					self.tracks[i].predict(detections[assignment[i]],contours[assignment[i]])
					self.cur_frame.append(i)
			
			else:
				unassigned_tracks.append(i)
				self.tracks[i].lost_frames += 1
				self.tracks[i].self_play
		



		del_tracks = []
		for i in range(len(self.tracks)):
			if self.tracks[i].lost_frames > self.max_frame_lost:
				del_tracks.append(i)

		if len(del_tracks) > 0:
			for i in range(len(del_tracks)):
				del self.tracks[i]
				del assignment[i]

		for i in range(len(detections)):
			if i not in assignment:
				track = Tracks(self.trackId,self.max_frame_lost)
				track.predict(detections[i],contours[i])
				self.trackId +=1
				self.tracks.append(track)
				self.cur_frame.append(len(self.tracks)-1)
		
		self.last_frame = self.cur_frame











		




import numpy as np
from sklearn.cluster import DBSCAN
import cv2
from scipy.optimize import linear_sum_assignment

from filterpy.kalman import KalmanFilter

class KalmanBoxTracker:
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    State space: [x, y, z, vx, vy, vz]
    Measurement space: [x, y, z]
    """
    def __init__(self, bbox_dict):
        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        # State transition matrix
        self.kf.F = np.array([
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        # Measurement function
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        # Initial state uncertainty
        self.kf.P[3:, 3:] *= 1000.  # High uncertainty in unobservable initial velocities
        self.kf.P *= 10.
        
        # Process noise and measurement noise
        self.kf.Q[-3:, -3:] *= 0.01 # Small process noise in velocity
        self.kf.R *= 0.1 # Measurement noise

        # Set initial state
        center = bbox_dict['center']
        self.kf.x[:3] = center.reshape((3, 1))
        
        self.bbox_dict = bbox_dict
        self.missed_frames = 0
        self.hit_streak = 0
        self.time_since_update = 0
        self.id = -1 # Set by PersonTracker

    def update(self, bbox_dict):
        """
        Updates the state vector with observed bbox_dict.
        """
        self.time_since_update = 0
        self.hit_streak += 1
        self.missed_frames = 0
        self.bbox_dict = bbox_dict
        center = bbox_dict['center']
        self.kf.update(center.reshape((3, 1)))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box center.
        """
        # If no update for a while, increase uncertainty of velocity
        if self.time_since_update > 0:
            self.hit_streak = 0
            
        self.kf.predict()
        self.time_since_update += 1
        
        # Update the stored dictionary's center to the predicted one
        # so drawing functions display the expected position during occlusion
        self.bbox_dict['center'] = self.kf.x[:3].flatten()
        return self.bbox_dict['center']
        
class PersonTracker:
    def __init__(self, refl_penalty_weight=1.5, distance_threshold=1.0):
        # DBSCAN parameters for pedestrian-like clusters
        self.eps = 0.15 # Extreme separation for groups
        self.min_samples = 2 # Very low because points are sparse in tiny clusters

        # Pedestrian dimension constraints (in meters)
        self.h_min, self.h_max = 0.5, 2.3
        self.w_min, self.w_max = 0.1, 2.5
        self.l_min, self.l_max = 0.1, 2.5

        # Tracking state
        self.next_track_id = 1
        self.tracks = [] # List of KalmanBoxTracker instances
        self.max_missed = 3 # Number of frames to keep a track alive without a match
        self.dist_thresh = distance_threshold # Maximum distance (in meters) to match a track to a detection
        
        # Frontier 1 parameter
        self.refl_penalty_weight = refl_penalty_weight

    def get_compensated_reflectivity(self, lidar_data):
        x, y, z = lidar_data[:, 0], lidar_data[:, 1], lidar_data[:, 2]
        intensity = lidar_data[:, 3]
        distance = np.sqrt(x**2 + y**2 + z**2)
        reflectivity = intensity * (distance**2)
        denom = (np.max(reflectivity) - np.min(reflectivity))
        if denom > 0:
            reflectivity = (reflectivity - np.min(reflectivity)) / denom
        return reflectivity

    def preprocess(self, pc):
        roi_mask = (pc[:, 0] > 0) & (pc[:, 0] < 40) & \
                   (pc[:, 1] > -10) & (pc[:, 1] < 10) & \
                   (pc[:, 2] > -2) & (pc[:, 2] < 2)
        pc = pc[roi_mask]
        ground_mask = pc[:, 2] > -1.5 
        pc = pc[ground_mask]
        refl = self.get_compensated_reflectivity(pc).reshape(-1, 1)
        pc = np.hstack((pc, refl))
        return pc

    def cluster(self, pc):
        if len(pc) == 0:
            return []
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(pc[:, :3])
        labels = clustering.labels_
        clusters = []
        for label in set(labels):
            if label == -1: continue
            clusters.append(pc[labels == label])
        return clusters

    def filter_pedestrians(self, clusters):
        pedestrians = []
        for i, cluster in enumerate(clusters):
            p_min = np.min(cluster[:, :3], axis=0)
            p_max = np.max(cluster[:, :3], axis=0)
            l, w, h = p_max - p_min
            avg_intensity = np.mean(cluster[:, 3])
            avg_reflectivity = np.mean(cluster[:, 4])
            
            is_pedestrian = (self.h_min < h < self.h_max) and \
                            (self.w_min < w < self.w_max) and \
                            (self.l_min < l < self.l_max)
            if is_pedestrian:
                pedestrians.append({
                    'points': cluster,
                    'bbox': (p_min, p_max),
                    'center': (p_min + p_max) / 2,
                    'avg_intensity': avg_intensity,
                    'avg_reflectivity': avg_reflectivity,
                    'score': 1.0,
                    'raw_box': np.concatenate([(p_min + p_max) / 2, p_max - p_min, [0]])
                })
        return pedestrians

    def track(self, current_pedestrians):
        """
        Data association using Kalman Filter predictions and Hungarian algorithm.
        """
        # Predict locations for all existing tracks
        predicted_centers = []
        for trk in self.tracks:
            pred = trk.predict()
            predicted_centers.append(pred)

        if not current_pedestrians:
            # No detections: increment missed frames for all tracks
            for t in self.tracks:
                t.missed_frames += 1
            self.tracks = [t for t in self.tracks if t.missed_frames <= self.max_missed]
            return [self._format_track(t) for t in self.tracks]

        if not self.tracks:
            # First frame or no existing tracks: create new trackers for all detections
            for ped in current_pedestrians:
                trk = KalmanBoxTracker(ped)
                trk.id = self.next_track_id
                self.next_track_id += 1
                self.tracks.append(trk)
            return [self._format_track(t) for t in self.tracks]

        # Compute cost matrix (Euclidean distance between PREDICTED centers and new detections)
        cost_matrix = np.zeros((len(self.tracks), len(current_pedestrians)))
        for i, trk in enumerate(self.tracks):
            for j, d in enumerate(current_pedestrians):
                spatial_cost = np.linalg.norm(predicted_centers[i] - d['center'])
                
                # Frontier 1: Reflectivity-Augmented Tracking Cost
                # Extract the stored reflectivity signature from the track and the new detection
                trk_refl = trk.bbox_dict.get('reflectivity', 0.0)
                det_refl = d.get('reflectivity', 0.0)
                
                # Calculate the appearance penalty (difference in material signature)
                # A multiplier of self.refl_penalty_weight ensures that wildly different materials
                # will add "virtual distance" to the cost, preventing ID swaps during tight occlusions.
                refl_penalty = abs(trk_refl - det_refl) * self.refl_penalty_weight
                
                cost_matrix[i, j] = spatial_cost + refl_penalty

        # Hungarian algorithm to find optimal assignments
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_detections = set(range(len(current_pedestrians)))

        # Update matched tracks
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < self.dist_thresh:
                # Valid match, update the Kalman filter with the true detection
                self.tracks[r].update(current_pedestrians[c])
                unmatched_tracks.remove(r)
                unmatched_detections.remove(c)

        # Handle unmatched tracks (increment missed_frames)
        for r in unmatched_tracks:
            self.tracks[r].missed_frames += 1

        # Handle unmatched detections (create new tracks)
        for c in unmatched_detections:
            ped = current_pedestrians[c]
            trk = KalmanBoxTracker(ped)
            trk.id = self.next_track_id
            self.next_track_id += 1
            self.tracks.append(trk)

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.missed_frames <= self.max_missed]

        # Return all active tracks in the expected dictionary format for the visualizer
        return [self._format_track(t) for t in self.tracks]

    def _format_track(self, trk):
        """Helper to convert KalmanBoxTracker back into the expected dict format"""
        # Ensure 'missed_frames' state is propagated properly so the visualizer can grey-out occluded tracks
        trk_dict = dict(trk.bbox_dict)
        trk_dict['id'] = trk.id
        trk_dict['missed_frames'] = trk.missed_frames
        return trk_dict

if __name__ == "__main__":
    print("Person Tracker script ready.")

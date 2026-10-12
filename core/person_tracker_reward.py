import numpy as np
from sklearn.cluster import DBSCAN
import copy
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

class KalmanBoxTrackerReward:
    """
    State-space: [x, y, z, vx, vy, vz]
    """
    def __init__(self, bbox_dict, num_bins=10):
        self.num_bins = num_bins
        # Initialise Kalman filter
        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        self.kf.F = np.array([
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000. 
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        self.id = 0
        self.time_since_update = 0
        self.hit_streak = 0
        self.missed_frames = 0
        
        self.bbox_dict = copy.deepcopy(bbox_dict)
        center = self.bbox_dict['center']
        self.kf.x[:3] = center.reshape((3, 1))

        # Store the reflectivity fingerprint
        self.refl_hist = self._extract_hist(bbox_dict)

    def _extract_hist(self, bbox_dict):
        hist = bbox_dict.get('refl_hist', None)
        if hist is not None:
            return hist
        
        # Fallback for evaluation scripts that only pass scalar reflectivity
        r_val = bbox_dict.get('reflectivity', 0.0)
        hist = np.zeros(self.num_bins)
        # Assuming r_val is normalized [0, 1]
        bin_idx = int(np.clip(r_val * self.num_bins, 0, self.num_bins - 1))
        hist[bin_idx] = 1.0
        return hist

    def update(self, bbox_dict):
        """
        Updates the state vector with observed bbox_dict.
        """
        self.time_since_update = 0
        self.hit_streak += 1
        self.missed_frames = 0
        self.bbox_dict = copy.deepcopy(bbox_dict)
        center = bbox_dict['center']
        self.kf.update(center.reshape((3, 1)))

        # Update running histogram (exponential moving average)
        new_hist = self._extract_hist(bbox_dict)
        self.refl_hist = 0.8 * self.refl_hist + 0.2 * new_hist
        # re-normalize to ensure it sums to 1
        hist_sum = np.sum(self.refl_hist)
        if hist_sum > 0:
            self.refl_hist /= hist_sum
        self.bbox_dict['refl_hist'] = self.refl_hist

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box center.
        """
        if self.time_since_update > 0:
            self.hit_streak = 0
            
        self.kf.predict()
        self.time_since_update += 1
        
        self.bbox_dict['center'] = self.kf.x[:3].flatten()
        return self.bbox_dict['center']

class PersonTrackerReward:
    def __init__(self, refl_weight=0.5, distance_threshold=1.0, sigma_spatial=0.5):
        # DBSCAN parameters for pedestrian-like clusters
        self.eps = 0.15 
        self.min_samples = 2 

        # Pedestrian dimension constraints (in meters)
        self.h_min, self.h_max = 0.5, 2.3
        self.w_min, self.w_max = 0.1, 2.5
        self.l_min, self.l_max = 0.1, 2.5

        # Tracking state
        self.next_track_id = 1
        self.tracks = [] 
        self.max_missed = 3 
        self.dist_thresh = distance_threshold 
        
        # Affinity weighting parameters
        self.refl_weight = refl_weight
        self.sigma_spatial = sigma_spatial
        self.num_bins = 10

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
            
            # Assuming reflectivity is concatenated as the 5th column
            reflectivity_vals = cluster[:, 4] if cluster.shape[1] > 4 else cluster[:, 3]
            avg_reflectivity = np.mean(reflectivity_vals)
            
            # Method 1: Compute Reflectivity Fingerprint (Normalized Histogram)
            hist, _ = np.histogram(reflectivity_vals, bins=self.num_bins, range=(0.0, 1.0), density=False)
            hist = hist.astype(float)
            hist_sum = np.sum(hist)
            if hist_sum > 0:
                hist /= hist_sum
            
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
                    'refl_hist': hist, # The fingerprint
                    'score': 1.0,
                    'raw_box': np.concatenate([(p_min + p_max) / 2, p_max - p_min, [0]])
                })
        return pedestrians

    def track(self, current_pedestrians):
        """
        Data association using Maximum-Weight Bipartite Matching (Similarity Inversion)
        """
        predicted_centers = []
        for trk in self.tracks:
            pred = trk.predict()
            predicted_centers.append(pred)

        if not current_pedestrians:
            for t in self.tracks:
                t.missed_frames += 1
            self.tracks = [t for t in self.tracks if t.missed_frames <= self.max_missed]
            return [self._format_track(t) for t in self.tracks]

        if not self.tracks:
            for ped in current_pedestrians:
                trk = KalmanBoxTrackerReward(ped, num_bins=self.num_bins)
                trk.id = self.next_track_id
                self.next_track_id += 1
                self.tracks.append(trk)
            return [self._format_track(t) for t in self.tracks]

        # Method 1: Construct inverted Cost Matrix from Composite Affinity
        cost_matrix = np.zeros((len(self.tracks), len(current_pedestrians)))
        
        for i, trk in enumerate(self.tracks):
            for j, d in enumerate(current_pedestrians):
                d_spatial = np.linalg.norm(predicted_centers[i] - d['center'])
                
                # Spatial gating to forbid physically impossible matches
                if d_spatial > self.dist_thresh:
                    cost_matrix[i, j] = 1000.0 # Extremely high cost
                    continue
                
                # 1. Spatial Affinity (Gaussian pull instead of linear push)
                s_spatial = np.exp(-(d_spatial**2) / (2 * (self.sigma_spatial**2)))
                
                # 2. Material Affinity (Bhattacharyya coefficient)
                trk_hist = trk.refl_hist
                det_hist = d.get('refl_hist', None)
                if det_hist is None:
                    # Fallback if evaluation script only passes scalar reflectivity
                    r_val = d.get('reflectivity', 0.0)
                    det_hist = np.zeros(self.num_bins)
                    bin_idx = int(np.clip(r_val * self.num_bins, 0, self.num_bins - 1))
                    det_hist[bin_idx] = 1.0

                s_refl = np.sum(np.sqrt(trk_hist * det_hist))
                
                # 3. Composite Affinity
                s_ij = (1.0 - self.refl_weight) * s_spatial + self.refl_weight * s_refl
                
                # 4. Similarity Inversion (Convert reward back to cost for scipy LAP solver)
                cost_matrix[i, j] = 1.0 - s_ij

        # Hungarian algorithm to find optimal assignments (minimizes inverted cost)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_detections = set(range(len(current_pedestrians)))

        # Update matched tracks
        for r, c in zip(row_ind, col_ind):
            # A valid match must pass the distance gate (cost < 1000)
            if cost_matrix[r, c] < 1000.0:
                self.tracks[r].update(current_pedestrians[c])
                unmatched_tracks.remove(r)
                unmatched_detections.remove(c)

        for r in unmatched_tracks:
            self.tracks[r].missed_frames += 1

        for c in unmatched_detections:
            ped = current_pedestrians[c]
            trk = KalmanBoxTrackerReward(ped, num_bins=self.num_bins)
            trk.id = self.next_track_id
            self.next_track_id += 1
            self.tracks.append(trk)

        self.tracks = [t for t in self.tracks if t.missed_frames <= self.max_missed]
        return [self._format_track(t) for t in self.tracks]

    def _format_track(self, trk):
        trk_dict = dict(trk.bbox_dict)
        trk_dict['id'] = trk.id
        trk_dict['missed_frames'] = trk.missed_frames
        return trk_dict


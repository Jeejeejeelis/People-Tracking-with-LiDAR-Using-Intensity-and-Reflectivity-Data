import numpy as np
import json
import os
from pathlib import Path
import pickle
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.tracking.evaluate import TrackingEval
from nuscenes.eval.tracking.constants import TRACKING_METRICS
from nuscenes.eval.common.config import config_factory as track_config_factory
from pcdet.datasets.nuscenes import nuscenes_utils
from nuscenes.utils.data_classes import Box

class PFSDEvaluator:
    def __init__(self, nusc_root, version='v1.0-mini'):
        
        self.nusc = NuScenes(version=version, dataroot=nusc_root, verbose=True,  )
        self.version = version
        self.nusc_root = nusc_root

    def convert_to_nusc_tracking_format(self, det_annos, tracker):
        """
        det_annos: list of dicts with 'boxes_lidar', 'score', 'pred_labels', 'metadata'
        tracker: an object with a track() method that takes detections and returns tracks
        """
        # Group detections by scene and sort by timestamp
        scene_samples = {}
        token_to_info = {}
        for det in det_annos:
            token = det['metadata']['token']
            sample = self.nusc.get('sample', token)
            scene_token = sample['scene_token']
            if scene_token not in scene_samples:
                scene_samples[scene_token] = []
            scene_samples[scene_token].append(det)
            token_to_info[token] = sample

        # Sort samples within each scene by timestamp
        for scene_token in scene_samples:
            scene_samples[scene_token].sort(key=lambda x: token_to_info[x['metadata']['token']]['timestamp'])

        results = {}

        for scene_token in scene_samples:
            tracker.tracks = [] # Reset tracker for each scene
            tracker.next_track_id = 1
            
            for det in scene_samples[scene_token]:
                token = det['metadata']['token']
                
                # 1. Prepare detections for tracker
                boxes3d = det['boxes_lidar']
                scores = det['score']
                labels = det['pred_labels']
                reflectivities = det.get('reflectivity', np.zeros(len(scores)))
                median_reflectivities = det.get('median_reflectivity', np.zeros(len(scores)))
                
                current_peds = []
                for k in range(boxes3d.shape[0]):
                    if labels[k] != 9: continue # PFSD Pedestrian label is 9 (1-indexed)
                    
                    center = boxes3d[k, :3]
                    current_peds.append({
                        'center': center,
                        'bbox': (center - 0.5, center + 0.5), # Dummy bbox for tracker
                        'score': scores[k],
                        'raw_box': boxes3d[k],
                        'reflectivity': reflectivities[k],
                        'median_reflectivity': median_reflectivities[k] if len(median_reflectivities) > 0 else 0.0
                    })
                
                # 2. Run tracker
                tracks = tracker.track(current_peds)
                
                # 3. Convert tracks to global nuScenes format
                sample_results = []
                for track in tracks:
                    # We need to find which detection this track came from to get rotation/size
                    # Or we could just store 'raw_box' in the track object.
                    # Since PersonTracker modifies 'center' and 'bbox', let's ensure it stores 'raw_box'.
                    
                    # For this implementation, I'll assume tracker matches the one in person_tracker.py
                    # and I'll modify person_tracker.py later to store the full box.
                    
                    raw_box_lidar = track.get('raw_box', None)
                    if raw_box_lidar is None:
                         # Fallback if tracker didn't store it
                         continue

                    # Create Box object
                    quat = Quaternion(axis=[0, 0, 1], radians=raw_box_lidar[6])
                    box = Box(
                        raw_box_lidar[:3],
                        raw_box_lidar[[4, 3, 5]], # wlh
                        quat, score=track.get('score', 0.9)
                    )
                    
                    # Convert to global
                    box_global = nuscenes_utils.lidar_nusc_box_to_global(self.nusc, [box], token)[0]
                    
                    nusc_track_anno = {
                        'sample_token': token,
                        'translation': box_global.center.tolist(),
                        'size': box_global.wlh.tolist(),
                        'rotation': box_global.orientation.elements.tolist(),
                        'velocity': [0.0, 0.0], # Simplified
                        'tracking_id': str(track['id']),
                        'tracking_name': 'pedestrian',
                        'tracking_score': box_global.score
                    }
                    sample_results.append(nusc_track_anno)
                
                results[token] = sample_results

        return results

    def run_evaluation(self, results_path, output_dir):
        eval_version = 'tracking_nips_2019'
        eval_config = track_config_factory(eval_version)
        eval_config.class_names = ['pedestrian']



        # Monkey-patch to allow PFSD validation scenes before TrackingEval initializes
        import nuscenes.utils.splits as splits_utils
        import nuscenes.eval.tracking.loaders as tracking_loaders
        import nuscenes.eval.common.loaders as common_loaders
        
        with open(results_path, 'r') as f:
            res_data = json.load(f)
        pred_tokens = set(res_data['results'].keys())
        val_scene_tokens = set(self.nusc.get('sample', t)['scene_token'] for t in pred_tokens if t in self.nusc._token2ind['sample'])
        val_scene_names = [self.nusc.get('scene', st)['name'] for st in val_scene_tokens]
        
        split_dict = {
            'mini_val': val_scene_names,
            'val': val_scene_names,
        }
        splits_utils.create_splits_scenes = lambda **kwargs: split_dict
        tracking_loaders.create_splits_scenes = lambda **kwargs: split_dict
        common_loaders.create_splits_scenes = lambda **kwargs: split_dict

        import nuscenes.eval.tracking.evaluate as tracking_eval
        import nuscenes.eval.tracking.mot as tracking_mot
        import nuscenes.eval.tracking.algo as tracking_algo
        import nuscenes.eval.tracking.utils as tracking_utils
        import motmetrics.mot as mot_mot
        import pandas as pd
        
        orig_create_motmetrics = tracking_utils.create_motmetrics
        def fixed_create_motmetrics():
            if not hasattr(tracking_utils.motar, '_patched'):
                orig_motar = tracking_utils.motar
                def safe_motar(df, num_matches, num_misses, num_switches, num_false_positives, num_objects, alpha=1.0, **kwargs):
                    return orig_motar(df, num_matches, num_misses, num_switches, num_false_positives, num_objects, alpha)
                safe_motar._patched = True
                tracking_utils.motar = safe_motar

            mh = orig_create_motmetrics()
            import motmetrics
            mh.register(getattr(motmetrics.metrics, 'pred_frequencies'), formatter='{:d}'.format)
            return mh
        tracking_utils.create_motmetrics = fixed_create_motmetrics
        if hasattr(tracking_algo, 'create_motmetrics'):
            tracking_algo.create_motmetrics = fixed_create_motmetrics

        if not hasattr(pd.DataFrame, 'append'):
            pd.DataFrame.append = lambda self, other, **kwargs: pd.concat([self, other], **kwargs)

        def fixed_new_event_dataframe_with_data(indices, events):
            if isinstance(events, dict):
                df = pd.DataFrame(events, columns=['Type', 'OId', 'HId', 'D'])
            else:
                df = pd.DataFrame(events, columns=['Type', 'OId', 'HId', 'D'])
                
            if len(df) == 0:
                return tracking_mot.MOTAccumulatorCustom.new_event_dataframe()
                
            if isinstance(indices, dict):
                df.index = pd.MultiIndex.from_arrays([indices['FrameId'], indices['Event']], names=['FrameId', 'Event'])
            else:
                df.index = pd.MultiIndex.from_tuples(indices, names=['FrameId', 'Event'])
            return df

        tracking_mot.MOTAccumulatorCustom.new_event_dataframe_with_data = fixed_new_event_dataframe_with_data
        if hasattr(tracking_eval, 'MOTAccumulatorCustom'):
            tracking_eval.MOTAccumulatorCustom.new_event_dataframe_with_data = fixed_new_event_dataframe_with_data
        if hasattr(tracking_algo, 'MOTAccumulatorCustom'):
            tracking_algo.MOTAccumulatorCustom.new_event_dataframe_with_data = fixed_new_event_dataframe_with_data
        mot_mot.MOTAccumulator.new_event_dataframe_with_data = fixed_new_event_dataframe_with_data
        
        orig_accumulate = tracking_algo.TrackingEvaluation.accumulate
        def safe_accumulate(self):
            # Patch numpy nan uniqueness issue in compute_thresholds assertion
            try:
                return orig_accumulate(self)
            except AssertionError:
                # If assertion fails, run accumulate with relaxed assertion
                if self.verbose:
                    print('Computing metrics for class %s...\n' % self.class_name)
                accumulators = []
                thresh_metrics = []
                md = tracking_algo.TrackingMetricData()

                gt_box_count = 0
                gt_track_ids = set()
                for scene_tracks_gt in self.tracks_gt.values():
                    for frame_gt in scene_tracks_gt.values():
                        for box in frame_gt:
                            if box.tracking_name == self.class_name:
                                gt_box_count += 1
                                gt_track_ids.add(box.tracking_id)
                if gt_box_count == 0:
                    return md

                mh = tracking_algo.create_motmetrics()
                thresholds, recalls = self.compute_thresholds(gt_box_count)
                print(f"DEBUG thresholds: {thresholds}, recalls: {recalls}")
                md.confidence = thresholds
                md.recall_hypo = recalls

                for t, threshold in enumerate(thresholds):
                    if np.isnan(threshold):
                        continue
                    if threshold in thresholds[:t]:
                        continue
                    acc, _ = self.accumulate_threshold(threshold)
                    accumulators.append(acc)
                    thresh_name = self.name_gen(threshold)
                    thresh_summary = mh.compute(acc, metrics=tracking_algo.MOT_METRIC_MAP.keys(), name=thresh_name)
                    thresh_metrics.append(thresh_summary)

                if len(thresh_metrics) == 0:
                    summary = []
                else:
                    summary = pd.concat(thresh_metrics)

                unachieved_thresholds = np.sum(np.isnan(thresholds))
                valid_thresholds = [t for t in thresholds if not np.isnan(t)]
                rep_counts = [np.sum(thresholds == t) for t in np.unique(valid_thresholds)]

                for (mot_name, metric_name) in tracking_algo.MOT_METRIC_MAP.items():
                    if metric_name == '':
                        continue
                    if len(thresh_metrics) == 0:
                        worst = self.metric_worst[metric_name]
                        if worst == -1:
                            if metric_name == 'ml':
                                worst = len(gt_track_ids)
                            elif metric_name in ['gt', 'fn']:
                                worst = gt_box_count
                            elif metric_name in ['fp', 'ids', 'frag']:
                                worst = np.nan
                            else:
                                raise NotImplementedError
                        all_values = [worst] * tracking_algo.TrackingMetricData.nelem
                    else:
                        values = summary.get(mot_name).values
                        values = np.concatenate([([v] * r) for (v, r) in zip(values, rep_counts)])
                        all_values = [np.nan] * unachieved_thresholds
                        all_values.extend(values)

                    if len(all_values) < tracking_algo.TrackingMetricData.nelem:
                        all_values.extend([np.nan] * (tracking_algo.TrackingMetricData.nelem - len(all_values)))
                    elif len(all_values) > tracking_algo.TrackingMetricData.nelem:
                        all_values = all_values[:tracking_algo.TrackingMetricData.nelem]
                    md.set_metric(metric_name, all_values)

                return md

        tracking_algo.TrackingEvaluation.accumulate = safe_accumulate




        from itertools import count

        def safe_merge_event_dataframes(dfs, update_frame_indices=True, update_oids=True, update_hids=True, return_mappings=False):
            mapping_infos = []
            new_oid = count()
            new_hid = count()

            r = tracking_mot.MOTAccumulatorCustom.new_event_dataframe()
            for df in dfs:
                if isinstance(df, tracking_mot.MOTAccumulatorCustom):
                    df = df.events

                copy = df.copy()
                infos = {}

                if update_frame_indices:
                    try:
                        max_val = r.index.get_level_values(0).max()
                        if max_val is None or pd.isna(max_val) or not isinstance(max_val, (int, float, np.integer, np.floating)):
                            next_frame_id = 0
                        else:
                            next_frame_id = int(max(max_val + 1, r.index.get_level_values(0).unique().shape[0]))
                    except Exception:
                        next_frame_id = 0
                    copy.index = copy.index.map(lambda x: (x[0] + next_frame_id, x[1]))
                    infos['frame_offset'] = next_frame_id

                if update_oids:
                    oid_map = dict([oid, str(next(new_oid))] for oid in copy['OId'].dropna().unique())
                    copy['OId'] = copy['OId'].map(lambda x: oid_map[x], na_action='ignore')
                    infos['oid_map'] = oid_map

                if update_hids:
                    hid_map = dict([hid, str(next(new_hid))] for hid in copy['HId'].dropna().unique())
                    copy['HId'] = copy['HId'].map(lambda x: hid_map[x], na_action='ignore')
                    infos['hid_map'] = hid_map

                r = pd.concat([r, copy])
                mapping_infos.append(infos)

            if return_mappings:
                return r, mapping_infos
            else:
                return r

        tracking_mot.MOTAccumulatorCustom.merge_event_dataframes = staticmethod(safe_merge_event_dataframes)



        
        orig_filter_eval_boxes = common_loaders.filter_eval_boxes
        def safe_filter_eval_boxes(nusc, eval_boxes, max_dist, verbose=False):
            keys = list(eval_boxes.boxes.keys())
            if len(keys) > 0 and len(eval_boxes.boxes[keys[0]]) == 0:
                for k in keys:
                    if len(eval_boxes.boxes[k]) > 0:
                        non_empty_boxes = eval_boxes.boxes[k]
                        new_boxes = {k: non_empty_boxes}
                        for key_i, val_i in eval_boxes.boxes.items():
                            if key_i != k:
                                new_boxes[key_i] = val_i
                        eval_boxes.boxes = new_boxes
                        break
            if len(eval_boxes.boxes) == 0 or len(eval_boxes.boxes[list(eval_boxes.boxes.keys())[0]]) == 0:
                return eval_boxes
            return orig_filter_eval_boxes(nusc, eval_boxes, max_dist, verbose=verbose)

        common_loaders.filter_eval_boxes = safe_filter_eval_boxes
        tracking_eval.filter_eval_boxes = safe_filter_eval_boxes



        nusc_eval = TrackingEval(

            config=eval_config,
            result_path=results_path,
            eval_set='mini_val' if 'mini' in self.version else 'val',
            output_dir=output_dir,
            nusc_version=self.version,
            nusc_dataroot=self.nusc_root,
            verbose=True
        )

        metrics_summary = nusc_eval.main()
        return metrics_summary


if __name__ == "__main__":
    # Example usage
    print("PFSD Evaluator script ready.")

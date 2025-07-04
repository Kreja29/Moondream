#!/usr/bin/env python3

import os
from pathlib import Path
import sys
import rospy
import cv2
import numpy as np
from scipy.interpolate import griddata
from typing import List, Dict, Tuple, Any, Optional
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import traceback
import torch
from transformers import AutoModelForCausalLM
import time
import open3d as o3d

sys.path.append('/workspace/src/3DGazeNet')  # Add project root to sys.path
sys.path.append('/workspace/src/3DGazeNet/demo')  # Add demo directory to sys.path

from demo.models.face_detector import FaceDetectorIF
from demo.models.gaze_predictor import GazePredictorHandler
from demo.utils import config as cfg

class DatasetHelper:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def get_id_list(self) -> List[str]:
        return [d for d in os.listdir(self.dataset_path) if d.startswith('ManiGaze_ID_') and os.path.isdir(os.path.join(self.dataset_path, d))]

    def get_session_path(self, user_id: str, session: str) -> str:
        id_part = user_id.split('_', 1)[1]
        return os.path.join(self.dataset_path, user_id, 'ManiGaze', id_part, session)

    def get_video_files(self, user_id: str, session: str, camera: str = 'kinect2') -> Tuple[Optional[str], Optional[str]]:
        session_path = self.get_session_path(user_id, session)
        rgb = os.path.join(session_path, camera, 'rgb.avi')
        depth = os.path.join(session_path, camera, 'depth.avi')
        return (rgb if os.path.exists(rgb) else None, depth if os.path.exists(depth) else None)

    def load_mouse_events(self, user_id: str, session: str) -> Optional[np.ndarray]:
        path = os.path.join(self.get_session_path(user_id, session), 'mouse_event.txt')
        return np.loadtxt(path, dtype=int) if os.path.exists(path) else None

    def load_gaze_labels(self, user_id: str, session: str) -> Optional[np.ndarray]:
        path = os.path.join(self.get_session_path(user_id, session), 'gaze_label.txt')
        return np.loadtxt(path, dtype=int) if os.path.exists(path) else None

    def load_speech_labels(self, user_id: str, session: str) -> Optional[List[str]]:
        path = os.path.join(self.get_session_path(user_id, session), 'speech_label.txt')
        return open(path).readlines() if os.path.exists(path) else None

    def load_left_arm(self, user_id: str, session: str) -> Optional[np.ndarray]:
        path = os.path.join(self.get_session_path(user_id, session), 'left_arm.txt')
        return np.loadtxt(path, delimiter=',') if os.path.exists(path) else None

    def load_right_arm(self, user_id: str, session: str) -> Optional[np.ndarray]:
        path = os.path.join(self.get_session_path(user_id, session), 'right_arm.txt')
        return np.loadtxt(path, delimiter=',') if os.path.exists(path) else None

class GazeEstimator:
    def __init__(self, 
                 model_ckpt_path=None, 
                 device=None, 
                 det_thresh=0.5, 
                 det_size=224):
        """
        model_ckpt_path: path to the 3DGazeNet checkpoint (.pth)
        device: 'cuda:0' or 'cpu' (auto if None)
        det_thresh: face detection threshold
        det_size: face detection input size
        """
        # Device selection
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.cfg = cfg
        self.cfg.PREDICTOR.NUM_LAYERS = 18
        self.cfg.PREDICTOR.BACKBONE_TYPE = 'resnet'
        if model_ckpt_path is None:
            base = os.path.dirname(os.path.abspath(__file__))
            #model_ckpt_path = os.path.join(base, 'data', 'checkpoints', 'res18_x128_all_vfhq_vert.pth')
            #model_ckpt_path = ('/workspace/src/3DGazeNet/data/models/imagenet/resnet18-5c106cde.pth')
            model_ckpt_path = ('/workspace/src/3DGazeNet/data/3dgazenet/models/singleview/vertex/ALL/test_0/checkpoint.pth')
        self.cfg.PREDICTOR.PRETRAINED = model_ckpt_path
        self.cfg.PREDICTOR.MODE = 'vertex'
        self.cfg.PREDICTOR.IMAGE_SIZE = [128, 128]
        self.cfg.PREDICTOR.NUM_POINTS_OUT_EYES = 962
        self.cfg.PREDICTOR.NUM_POINTS_OUT_FACE = 68
        self.cfg.DEVICE = device
        self.face_detector = FaceDetectorIF(det_thresh, det_size)
        self.gaze_predictor = GazePredictorHandler(self.cfg.PREDICTOR, device=device)

    def predict(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        bboxs, kpts, faces = self.face_detector.run(frame_rgb)
        if kpts is None or len(kpts) == 0:
            return None
        areas = [np.prod(bbox[2:4] - bbox[0:2]) for bbox in bboxs]
        idx = int(np.argmax(areas))
        kpt5 = kpts[idx]
        left_eye = kpt5[0]
        right_eye = kpt5[1]
        cx = (left_eye[0] + right_eye[0]) / 2.0
        cy = (left_eye[1] + right_eye[1]) / 2.0

        h, w = frame_rgb.shape[:2]
        eye_center = np.array([cx / w, cy / h])  # Normalize to [0, 1] range
        with torch.no_grad():
            result = self.gaze_predictor(frame_rgb, kpt5, undo_roll=True)
        return eye_center, result.get('gaze_combined', None)
    

class GazeDetectionEvaluator:
    def __init__(self):
        # Parameters for dataset and evaluation
        self.dataset_dir = rospy.get_param('~dataset_dir', '/workspace/dataset')
        self.results_dir = rospy.get_param('~results_dir', '/workspace/results')
        self.bridge = CvBridge()
        self.processing_times = []
        self.processed_count = 0

        # R_pos and T_pose are from camera to rgb
        # Rotation matrix
        self.R_pos = np.array([
            [1.0,  0.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0]
        ])
        self.R_pos_new = np.array([
            [0.0,  -1.0,  0.0],
            [-1.0, 0.0,  0.0],
            [0.0,  0.0, -1.0]
        ])

        # Translation vector
        self.T_pos = np.array([
            [0.0],
            [0.0],
            [1.0]
        ])
        # Transformation from robot to Kinect camera
        R_robot_to_kinect = np.array([
        [ 0.06456814,  0.99782479,  0.01329078],
        [-0.32914885,  0.00872184,  0.94423777],
        [ 0.94206793, -0.06534232,  0.32899604]
        ])

        T_robot_to_kinect = np.array([
        [-0.10617324],
        [ 0.03664513],
        [-0.2154056 ]
        ])

        def transform_robot_to_camera_depth(self, effector_pos: np.ndarray) -> np.ndarray:
            """
            Transform a point in robot coordinates to camera DEPTH coordinates.
            effector_pos: 3D point in robot coordinates (shape: (3,))
            Returns the transformed point in camera DEPTH coordinates.
            """
            effector_kinect = R_robot_to_kinect @ effector_pos + T_robot_to_kinect.flatten()
            effector_depth = self.R_pos @ effector_kinect.reshape(3, 1) + self.T_pos
            return effector_depth.flatten()
        

        # Marker positions in Kinect camera coordinates
        self.marker_positions_kinect = np.array([
            [ 0.08278523,  0.63388652,  0.81159113],
            [ 0.44042943, -0.16310945,  0.68141465],
            [ 0.24584326, -0.15143633,  0.69731402],
            [ 0.0531153 , -0.14831255,  0.71382785],
            [-0.14460588, -0.14245166,  0.73133182],
            [-0.33651502, -0.13401367,  0.75201091],
            [ 0.32852781, -0.20895353,  0.54134292],
            [ 0.13070989, -0.2012552 ,  0.55711913],
            [-0.05960627, -0.19672879,  0.57207172],
            [-0.25593353, -0.19064882,  0.59256937],
            [ 0.4116894 , -0.25854817,  0.38640783],
            [ 0.21406238, -0.25249177,  0.40218933],
            [ 0.02388491, -0.24782869,  0.4160821 ],
            [-0.16872942, -0.24101171,  0.43772051],
            [-0.36351209, -0.23879948,  0.45211431],
        ])
        # Precompute marker positions in RGB coordinate system
        self.marker_positions_human = np.array([
            (self.R_pos_new @ marker.reshape(3, 1) + self.T_pos).flatten()
            for marker in self.marker_positions_kinect
        ])

        self.marker_positions_camera_depth = np.array([
            (self.R_pos @ marker.reshape(3, 1) + self.T_pos).flatten()
            for marker in self.marker_positions_kinect
        ])
        
        rospy.loginfo(f"Marker positions in 'human' coordinates: {self.marker_positions_human}") #temp
        rospy.loginfo(f"Marker positions in camera DEPTH coordinates: {self.marker_positions_camera_depth}") #temp

        # Depth intrinsics
        self.K_d = np.array([
            [374.29986435, 0.0, 259.08931589],
            [0.0, 374.93952842, 221.61052956],
            [0.0, 0.0, 1.0]
        ])

        # RGB intrinsics
        self.K_rgb = np.array([
            [1099.89415734, 0.0, 973.3031593],
            [0.0, 1100.774053, 556.2757212],
            [0.0, 0.0, 1.0]
        ])

        # Extrinsics (depth → RGB)
        self.R_extr = np.array([
            [0.9997907489813892,  0.0039233244105278, -0.0200764981210007],
            [-0.0040864364506259,  0.9999589259368433, -0.0080899614566265],
            [0.0200439339543859,  0.0081703099576737,  0.999765715928901]
        ])
        self.T_extr = np.array([[0.0510585959051953], [-0.0014484138682882], [0.0117507879232922]])


        # Initialize model
        self.model = GazeEstimator()

        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)

        # Initialize dataset helper
        self.dataset_helper = DatasetHelper(self.dataset_dir)

        rospy.loginfo("GazeDetectionEvaluator initialized")

    def evaluate_dataset(self):
        rospy.loginfo(f"Evaluating dataset in {self.dataset_dir}")
        id_list = self.dataset_helper.get_id_list()
        session_order = ['MT', 'ET_center', 'OM1']
        session_order = ['ET_center', 'MT']
        total_processed = 0
        for user_id in id_list[:]: 
            #if not user_id == 'ManiGaze_ID_17':
            #    continue
            
            rospy.loginfo(f"Processing {user_id}")
            for session in session_order:
                session_path = self.dataset_helper.get_session_path(user_id, session)
                if not os.path.exists(session_path):
                    rospy.logwarn(f"  Session {session} not found for {user_id}")
                    continue
                rospy.loginfo(f"  Processing session {session}")
                if session == 'MT':
                    self.process_mt_session(user_id, session)
                elif session == 'ET_center':
                    self.process_et_center_session(user_id, session)
                elif session == 'OM1':
                    self.process_om_session(user_id, session)
                else:
                    rospy.logwarn(f"  Unknown session type: {session}")
                    continue
                total_processed += 1
        rospy.loginfo(f"Finished evaluating {total_processed} sessions.")

    def process_et_center_session(self, user_id: str, session: str):
        rgb_file, depth_file = self.dataset_helper.get_video_files(user_id, session)
        mouse_events = self.dataset_helper.load_mouse_events(user_id, session)
        speech_labels = self.dataset_helper.load_speech_labels(user_id, session)
        left_arm = self.dataset_helper.load_left_arm(user_id, session)
        right_arm = self.dataset_helper.load_right_arm(user_id, session)
        # Log loaded data shapes
        rospy.loginfo(f"    RGB file: {rgb_file}")
        rospy.loginfo(f"    Depth file: {depth_file}")
        if mouse_events is not None:
            rospy.loginfo(f"    Mouse events: {mouse_events.shape}")
        if speech_labels is not None:
            rospy.loginfo(f"    Speech labels: {len(speech_labels)} lines")
        if left_arm is not None:
            rospy.loginfo(f"    Left arm: {left_arm.shape}")
        if right_arm is not None:
            rospy.loginfo(f"    Right arm: {right_arm.shape}")
        # TODO: Add ET_center-specific evaluation logic here

    def process_mt_session(self, user_id: str, session: str):
        rgb_file, depth_file = self.dataset_helper.get_video_files(user_id, session)
        mouse_events = self.dataset_helper.load_mouse_events(user_id, session)
        gaze_labels = self.dataset_helper.load_gaze_labels(user_id, session)
        speech_labels = self.dataset_helper.load_speech_labels(user_id, session)
        rospy.loginfo(f"    RGB file: {rgb_file}")
        rospy.loginfo(f"    Depth file: {depth_file}")
        if mouse_events is not None:
            rospy.loginfo(f"    Mouse events: {mouse_events.shape}")
        if gaze_labels is not None:
            rospy.loginfo(f"    Gaze labels: {gaze_labels.shape}")
        if speech_labels is not None:
            rospy.loginfo(f"    Speech labels: {len(speech_labels)} lines")
        if rgb_file is None or not os.path.exists(rgb_file):
            rospy.logwarn(f"    RGB video file not found: {rgb_file}")
            return
        if mouse_events is None or gaze_labels is None:
            rospy.logwarn(f"    Mouse events or gaze labels not found for {user_id} {session}")
            return
        cap = cv2.VideoCapture(rgb_file)
        cap_depth = cv2.VideoCapture(depth_file)
        distance_errors = []
        angle_errors = []
        model_times = []
        calc_times = []
        frames_read_count = 0
        gaze_detected_count = 0
        # Prepare error log file in results directory
        error_log_path = os.path.join(self.results_dir, f"{user_id}_{session}_frame_errors_3DGazeNet.txt")
        with open(error_log_path, "w") as f:
            # Write header
            f.write("frame_idx,error_distance,error_angle,marker_idx,model_time,calc_time\n")
            for idx, event in enumerate(mouse_events):
                if idx < 1:  # Skip first 3000 frames
                    continue
                if event == -1:
                    continue
                rospy.loginfo(f"Processing frame {idx} for {user_id} {session}")
                # Set video to the correct frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    rospy.logwarn(f"    Could not read frame {idx} from {rgb_file}")
                    continue
                cap_depth.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret_depth, depth_frame = cap_depth.read()
                if not ret_depth:
                    rospy.logwarn(f"    Could not read depth frame {idx} from {depth_file}")
                    continue
                
                frames_read_count += 1

                # Reconstruct a correct depth frame
                if depth_frame is not None:
                    depth_m = self.reconstruct_depth_16bit(depth_frame)
                else:
                    depth_m = None

                # --- Model (gaze) processing time ---
                t0 = time.time()
                # Get gaze from frame using 3DGazeNet model
                # eye_center is normalized (0, 1) coordinates of the eye center in the RGB image
                # gaze_vec3d is a 3D gaze vector in camera RGB coordinates

                result = self.model.predict(frame)
                if result is None:
                    rospy.logwarn(f"    No gaze or eye center detected in frame {idx}")
                    continue
                eye_center, gaze_vec3d = result

                eye_center_u = eye_center[0]  # normalized u coordinate
                eye_center_v = eye_center[1]  # normalized v coordinate

                if gaze_vec3d is None:
                    rospy.logwarn(f"    No gaze detected in frame {idx}")
                    continue
                if eye_center is None:
                    rospy.logwarn(f"    No eye center detected in frame {idx}")
                    continue
                
                t1 = time.time()
                
                gaze_detected_count += 1
                model_times.append(t1 - t0)

                rospy.loginfo(f"    gaze {gaze_vec3d}")
                rospy.loginfo(f"    model processing time: {t1 - t0:.3f}s")

                t2 = time.time()
                # Convert eye_center to 3d coordinates in camera DEPTH frame
                eye_center_pred_3d_depth, gaze_point_3d_depth = self.find_3d_point_from_rgb_gaze(
                    eye_center_u, eye_center_v, depth_m,
                    self.K_rgb, self.K_d, self.R_extr, self.T_extr,
                    rgb_shape=frame.shape,
                    gaze_vec3d_rgb=gaze_vec3d,
                    visualize=True  # Set to True to visualize the 3D points and line
                ) if depth_m is not None else (None, None)
                
                if eye_center_pred_3d_depth is None or gaze_point_3d_depth is None:
                    rospy.logwarn(f"    No valid 3D eye center or gaze point found in frame {idx}")
                    continue

                # Get marker index from gaze_labels
                marker_idx = gaze_labels[idx]
                if marker_idx < 0 or marker_idx >= self.marker_positions_kinect.shape[0]:
                    rospy.logwarn(f"    Invalid marker index {marker_idx} at frame {idx}")
                    continue
                # Marker position in depth camera coordinate system
                marker_camera = self.marker_positions_camera_depth[marker_idx]
                rospy.loginfo(f"    Marker {marker_idx} position in camera (depth) coordinates: {marker_camera}")

                error_distance, error_angle = self.get_marker_errors(eye_center_pred_3d_depth, gaze_point_3d_depth, marker_camera)
                t3 = time.time()
                calc_times.append(t3 - t2)
                
                # Append errors to lists
                distance_errors.append(error_distance)
                angle_errors.append(error_angle)

                # Log all error components and timing for this frame
                rospy.loginfo(f"    Frame {idx}: error={error_distance:.3f}m, marker={marker_idx}, model_time={t1-t0:.3f}s, calc_time={t3-t2:.3f}s")
                # Write per-frame error to file
                f.write(f"{idx},{error_distance:.6f},{error_angle:.6f},{marker_idx},{t1-t0:.6f},{t3-t2:.6f}\n")
        cap.release()
        cap_depth.release()

        gaze_detection_rate = (gaze_detected_count / frames_read_count * 100) if frames_read_count > 0 else 0

        if distance_errors and angle_errors and model_times and calc_times:
            # Compute and log mean and std for all error components and timings
            mean_error_distance = np.mean(distance_errors)
            std_error_distance = np.std(distance_errors)
            mean_error_angle = np.mean(angle_errors)
            std_error_angle = np.std(angle_errors)
            mean_model_time = np.mean(model_times)
            std_model_time = np.std(model_times)
            mean_calc_time = np.mean(calc_times)
            std_calc_time = np.std(calc_times)
            rospy.loginfo(f"    Gaze detection rate: {gaze_detection_rate:.2f}% ({gaze_detected_count}/{frames_read_count})")
            rospy.loginfo(f"    MT session mean distance error: {mean_error_distance:.3f}±{std_error_distance:.3f}m over {len(distance_errors)} frames")
            rospy.loginfo(f"    MT session mean angle error: {mean_error_angle:.3f}±{std_error_angle:.3f}m over {len(angle_errors)} frames")
            rospy.loginfo(f"    Model time: {mean_model_time:.3f}±{std_model_time:.3f}s")

            summary_log_path = os.path.join(self.results_dir, f"{user_id}_{session}_summary_3DGazeNet.txt")
            with open(summary_log_path, "w") as f:
                summary_line = (
                    f"Summary: "
                    f"GazeDetectionRate={gaze_detection_rate:.2f}%, "
                    f"FramesWithGaze={gaze_detected_count}, "
                    f"FramesProcessed={frames_read_count}, "
                    f"MeanDistanceError={mean_error_distance:.6f}, StdDistanceError={std_error_distance:.6f}, "
                    f"MeanAngleError={mean_error_angle:.6f}, StdAngleError={std_error_angle:.6f}, "
                    f"MeanModelTime={mean_model_time:.6f}, StdModelTime={std_model_time:.6f}, "
                    f"MeanCalcTime={mean_calc_time:.6f}, StdCalcTime={std_calc_time:.6f}\n"
                )
                f.write(summary_line)
        else:
            rospy.loginfo(f"    No valid frames processed for MT session")
            rospy.loginfo(f"    Gaze detection rate: {gaze_detection_rate:.2f}% ({gaze_detected_count}/{frames_read_count})")
            summary_log_path = os.path.join(self.results_dir, f"{user_id}_{session}_summary.txt")
            with open(summary_log_path, "w") as f:
                summary_line = (
                    f"Summary: "
                    f"GazeDetectionRate={gaze_detection_rate:.2f}%, "
                    f"FramesWithGaze={gaze_detected_count}, "
                    f"FramesProcessed={frames_read_count}, "
                    f"No frames with successful 3D correspondence.\n"
                )
                f.write(summary_line)

    def process_om_session(self, user_id: str, session: str):
        rgb_file, depth_file = self.dataset_helper.get_video_files(user_id, session)
        speech_labels = self.dataset_helper.load_speech_labels(user_id, session)
        # Log loaded data shapes
        rospy.loginfo(f"    RGB file: {rgb_file}")
        rospy.loginfo(f"    Depth file: {depth_file}")
        if speech_labels is not None:
            rospy.loginfo(f"    Speech labels: {len(speech_labels)} lines")
        # TODO: Add OM-specific evaluation logic here

    def get_depth_at_pixel(self, depth_frame: Optional[np.ndarray], u_norm: float, v_norm: float, filtering: bool = True, filter_width: int = 3) -> Optional[float]:
        # Extract the depth value from the depth frame at normalized coordinates (u_norm, v_norm)
        if depth_frame is None:
            return None
        h, w = depth_frame.shape[:2]
        x = min(int(u_norm * w), w - 1)
        y = min(int(v_norm * h), h - 1)
        z = depth_frame[y, x]
        if filtering:
            half = filter_width // 2
            # Do not apply filtering if filter would be outside of image
            if (x - half) < 0 or (x + half) > (w - 1):
                filtering = False
            elif (y - half) < 0 or (y + half) > (h - 1):
                filtering = False
            # Return the filtered depth value of the pixel neighborhood
            if filtering:
                z_list = depth_frame[y - half : y + half + 1, x - half : x + half + 1]
                z = np.median(z_list)
        if z <= 0:
            rospy.logwarn("Warning! Depth is zero!")
        return float(z)
    
    @staticmethod
    def depth_to_points(depth, fx, fy, cx, cy): # depth camera intrinsics
        h, w = depth.shape
        i, j = np.indices((h, w))
        z = depth.astype(np.float32)
        x = (j - cx) * z / fx
        y = (i - cy) * z / fy
        return np.stack((x, y, z), axis=-1)  # shape: (H, W, 3)

    @staticmethod
    def transform_to_rgb_frame(points, R, T): # depth to rgb camera frame
        points_flat = points.reshape(-1, 3).T  # shape: (3, N)
        transformed = R @ points_flat + T  # shape: (3, N)
        return transformed.T.reshape(points.shape)
    
    @staticmethod
    def project_to_image(points, fx, fy, cx, cy, rgb_shape): # rgb camera intrinsics
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        u = (x * fx / z + cx).astype(np.int32)
        v = (y * fy / z + cy).astype(np.int32)

        mask = (z > 0) & (u >= 0) & (u < rgb_shape[1]) & (v >= 0) & (v < rgb_shape[0])
        return u, v, mask

    def align_depth_to_rgb(self, depth, rgb_shape, intrinsics_d, intrinsics_rgb, R, T): # depth to rgb camera frame
        fx_d, fy_d, cx_d, cy_d = intrinsics_d
        fx_rgb, fy_rgb, cx_rgb, cy_rgb = intrinsics_rgb

        points = GazeDetectionEvaluator.depth_to_points(depth, fx_d, fy_d, cx_d, cy_d)
        points_rgb = GazeDetectionEvaluator.transform_to_rgb_frame(points, R, T)
        u, v, mask = GazeDetectionEvaluator.project_to_image(points_rgb, fx_rgb, fy_rgb, cx_rgb, cy_rgb, rgb_shape)

        aligned_depth = np.zeros(rgb_shape[:2], dtype=np.float32)
        aligned_depth[v[mask], u[mask]] = points_rgb[..., 2][mask]
        return aligned_depth

    def fill_empty_pixels(self, depth_map, method='linear'):
        """
        Fill empty (zero) pixels in a depth map using interpolation.

        Parameters:
            depth_map (np.ndarray): 2D array with depth values, 0 for missing.
            method (str): 'linear', 'nearest', or 'cubic'. 'linear' recommended.

        Returns:
            np.ndarray: depth map with missing values filled.
        """
        h, w = depth_map.shape
        mask = depth_map > 0

        if np.count_nonzero(mask) < 100:
            rospy.logwarn("Warning: Too few valid points to interpolate.")
            return depth_map

        # Get valid pixel locations and their depth values
        y_coords, x_coords = np.where(mask)
        valid_points = np.stack((x_coords, y_coords), axis=-1)
        valid_values = depth_map[y_coords, x_coords]

        # Prepare grid of all pixel locations
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        grid_points = np.stack((grid_x.ravel(), grid_y.ravel()), axis=-1)

        # Interpolate
        interpolated = griddata(
            points=valid_points,
            values=valid_values,
            xi=grid_points,
            method=method,
            fill_value=0  # fill unresolvable pixels with 0 again
        )

        return interpolated.reshape((h, w)).astype(depth_map.dtype)

    @staticmethod
    def reconstruct_depth_16bit(depth_frame: np.ndarray) -> np.ndarray:
        """
        Reconstruct 16-bit depth values from an 8-bit 3-channel depth frame and convert to meters.
        Assumes depth_frame is an 8-bit 3-channel image where:
        - channel 0: low byte
        - channel 1: high byte
        Returns a 2D np.ndarray of dtype np.float32, in meters.
        """
        low_byte = depth_frame[:, :, 0]
        high_byte = depth_frame[:, :, 1]
        depth_16bit = (high_byte.astype(np.uint16) << 8) | low_byte.astype(np.uint16)
        depth_m = depth_16bit.astype(np.float32) / 1000.0  # convert to meters
        return depth_m

    def find_3d_point_from_rgb_gaze(self, u_norm, v_norm, depth_frame, K_rgb, K_d, R_extr, T_extr, rgb_shape, gaze_vec3d_rgb, visualize=False):
        """
        Given normalized (u,v) in RGB image and a 3D gaze vector in RGB camera coordinates, find:
        - The closest 3D point in the depth camera frame to the backprojected gaze ray (eye center in depth)
        - Convert this point to RGB camera coordinates (eye center in RGB)
        - Subtract the gaze vector (in RGB) to get the gaze point in RGB
        - Convert this gaze point back to depth camera coordinates (gaze point in depth)
        Returns (eye_center_pred_3d_rgb, gaze_point_3d_depth)
        """
        if depth_frame is None:
            return None, None
        h_d, w_d = depth_frame.shape[:2]
        h_rgb, w_rgb = rgb_shape[:2]
        u = u_norm * w_rgb
        v = v_norm * h_rgb
        fx_rgb, fy_rgb, cx_rgb, cy_rgb = K_rgb[0,0], K_rgb[1,1], K_rgb[0,2], K_rgb[1,2]
        # Two points along the ray in RGB camera frame (z=0 and z=1)
        p0_rgb = np.array([(u - cx_rgb) / fx_rgb * 0, (v - cy_rgb) / fy_rgb * 0, 0])
        p1_rgb = np.array([(u - cx_rgb) / fx_rgb * 1, (v - cy_rgb) / fy_rgb * 1, 1])
        # Transform to depth camera frame
        R = R_extr
        T = T_extr.flatten()
        p0_depth = R.T @ (p0_rgb - T)
        p1_depth = R.T @ (p1_rgb - T)
        # Backproject all valid depth pixels to 3D in depth camera frame
        fx_d, fy_d, cx_d, cy_d = K_d[0,0], K_d[1,1], K_d[0,2], K_d[1,2]
        i, j = np.indices((h_d, w_d))
        z = depth_frame.astype(np.float32)
        x = (j - cx_d) * z / fx_d
        y = (i - cy_d) * z / fy_d
        points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        valid = (z > 0).reshape(-1)
        points = points[valid]
        if points.shape[0] == 0:
            return None, None
        # Find closest point to the line
        line_vec = p1_depth - p0_depth
        line_vec /= np.linalg.norm(line_vec)
        vecs = points - p0_depth
        t = np.dot(vecs, line_vec)
        proj = p0_depth + np.outer(t, line_vec)
        dists = np.linalg.norm(points - proj, axis=1)
        idx = np.argmin(dists)
        eye_center_pred_3d_depth = points[idx]
        # Convert eye center from depth to RGB camera coordinates
        eye_center_pred_3d_rgb = (R_extr @ eye_center_pred_3d_depth.reshape(3, 1) + T_extr.reshape(3, 1)).flatten()
        # Subtract gaze vector in RGB to get gaze point in RGB
        gaze_point_3d_rgb = eye_center_pred_3d_rgb - gaze_vec3d_rgb
        # Convert gaze point from RGB to depth camera coordinates
        gaze_point_3d_depth = R_extr.T @ (gaze_point_3d_rgb - T_extr.flatten())
        if visualize:
            # Depth points as point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[::10])
            pcd.paint_uniform_color([0.2, 0.2, 1.0])
            t_line = np.linspace(-0.1, 2, 100)
            line_pts = p0_depth[None,:] + t_line[:,None] * line_vec[None,:]
            line_pcd = o3d.geometry.PointCloud()
            line_pcd.points = o3d.utility.Vector3dVector(line_pts)
            line_pcd.paint_uniform_color([1.0, 0.0, 0.0])
            sphere_eye = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere_eye.translate(eye_center_pred_3d_depth)
            sphere_eye.paint_uniform_color([0.0, 1.0, 0.0])
            sphere_gaze = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere_gaze.translate(gaze_point_3d_depth)
            sphere_gaze.paint_uniform_color([1.0, 0.0, 0.0])
            # Add a line between sphere_eye and sphere_gaze
            points_line = [eye_center_pred_3d_depth, gaze_point_3d_depth]
            lines = [[0, 1]]
            colors = [[0.0, 1.0, 0.0]]  # green line
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points_line)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            marker_meshes = []
            if self.marker_positions_camera_depth is not None:
                for m in self.marker_positions_camera_depth:
                    marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                    marker.translate(m)
                    marker.paint_uniform_color([1.0, 0.5, 0.0])
                    marker_meshes.append(marker)
            o3d.visualization.draw_geometries([pcd, line_pcd, sphere_eye, sphere_gaze, line_set] + marker_meshes)
        return eye_center_pred_3d_rgb, gaze_point_3d_depth
    
    def get_marker_errors(self, eyes, gaze_point_3d, marker):
        """
        Calculate the error between the predicted gaze line and the marker position.
        eyes: 3d coordinates of the eyes in depth camera coordinates
        gaze_point_3d: 3d coordinates of a point in the gaze line in depth camera coordinates
        marker: 3d coordinates of a point from which the error is calculated
        Returns: error distance and error angle
        """

        eyes = np.array(eyes)
        gaze = np.array(gaze_point_3d)
        marker = np.array(marker)

        # Vector from eyes to gaze and eyes to marker
        gaze_vector = gaze - eyes
        marker_vector = marker - eyes

        # Distance from point to line: ||(marker_vector x gaze_vector)|| / ||gaze_vector||
        cross_product = np.cross(marker_vector, gaze_vector)
        error_distance = np.linalg.norm(cross_product) / np.linalg.norm(gaze_vector)

        # Angle error: angle between gaze vector and marker vector
        dot_product = np.dot(gaze_vector, marker_vector)
        norms_mult = np.linalg.norm(gaze_vector) * np.linalg.norm(marker_vector)
        cos_theta = np.clip(dot_product / norms_mult, -1.0, 1.0)
        angle_error = np.arccos(cos_theta)  # in radians
        angle_error_degrees = np.degrees(angle_error)  # convert to degrees

        return error_distance, angle_error_degrees

if __name__ == "__main__":
    try:
        rospy.init_node('gaze_detection_evaluator', anonymous=True)
        evaluator = GazeDetectionEvaluator()
        evaluator.evaluate_dataset()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
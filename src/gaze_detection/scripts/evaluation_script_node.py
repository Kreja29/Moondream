#!/usr/bin/env python3

import os
from pathlib import Path
import sys
import rospy
import cv2
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import traceback
import torch
from transformers import AutoModelForCausalLM

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

class GazeDetectionEvaluator:
    def __init__(self):
        # Parameters for dataset and evaluation
        self.dataset_dir = rospy.get_param('~dataset_dir', '/workspace/dataset')
        self.results_dir = rospy.get_param('~results_dir', '/workspace/results')
        self.model_id = rospy.get_param('~model_id', "vikhyatk/moondream2")
        self.model_revision = rospy.get_param('~model_revision', "2025-01-09")
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

        # Translation vector
        self.T_pos = np.array([
            [0.0],
            [0.0],
            [1.0]
        ])
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
        self.marker_positions_rgb = np.array([
            (self.R_pos @ marker.reshape(3, 1) + self.T_pos).flatten()
            for marker in self.marker_positions_kinect
        ])

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
        self.model = self.initialize_model()
        if self.model is None:
            rospy.logerr("Failed to initialize Moondream 2 model. Exiting.")
            sys.exit(1)

        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)

        # Initialize dataset helper
        self.dataset_helper = DatasetHelper(self.dataset_dir)

        rospy.loginfo("GazeDetectionEvaluator initialized")

    def evaluate_dataset(self):
        rospy.loginfo(f"Evaluating dataset in {self.dataset_dir}")
        id_list = self.dataset_helper.get_id_list()
        session_order = ['MT', 'ET_center', 'OM1']
        total_processed = 0
        for user_id in id_list:
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
        errors = []
        for idx, event in enumerate(mouse_events):
            if event == -1:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                rospy.logwarn(f"    Could not read frame {idx} from {rgb_file}")
                continue
            cap_depth.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret_depth, depth_frame = cap_depth.read()
            if not ret_depth:
                rospy.logwarn(f"    Could not read depth frame {idx} from {depth_file}")
                depth_frame = None
            # Process frame to get gaze (2D, normalized 0-1)
            gaze_2d = self.get_gaze_from_frame(frame)
            if gaze_2d is None:
                rospy.logwarn(f"    No gaze detected in frame {idx}")
                continue
            u_norm, v_norm = gaze_2d  # normalized [0, 1]
            # Get depth at (u_norm, v_norm) from depth frame
            depth = self.get_depth_at_pixel(depth_frame, u_norm, v_norm)
            if depth is None:
                rospy.logwarn(f"    No depth for frame {idx}, normalized ({u_norm},{v_norm})")
                continue
            # Project to 3D in image coordinate system
            x_img = (u_norm * depth_frame.shape[1] - self.K_d[0, 2]) * depth / self.K_d[0, 0]
            y_img = (v_norm * depth_frame.shape[0] - self.K_d[1, 2]) * depth / self.K_d[1, 1]
            z_img = depth
            pt_img = np.array([[x_img], [y_img], [z_img]])
            # Convert to RGB camera coordinate system using extrinsics
            pt_rgb = self.R_extr @ pt_img + self.T_extr
            pred_3d_rgb = pt_rgb.flatten()
            # Get marker index from gaze_labels
            marker_idx = gaze_labels[idx]
            if marker_idx < 0 or marker_idx >= self.marker_positions_kinect.shape[0]:
                rospy.logwarn(f"    Invalid marker index {marker_idx} at frame {idx}")
                continue
            # Marker position in RGB coordinate system (precomputed)
            marker_rgb = self.marker_positions_rgb[marker_idx]
            # Compute error in RGB coordinate system
            error = np.linalg.norm(pred_3d_rgb - marker_rgb)
            errors.append(error)
            rospy.loginfo(f"    Frame {idx}: error={error:.3f}m, marker={marker_idx}")
        cap.release()
        cap_depth.release()
        if errors:
            mean_error = np.mean(errors)
            rospy.loginfo(f"    MT session mean error: {mean_error:.3f}m over {len(errors)} frames")
        else:
            rospy.loginfo(f"    No valid frames processed for MT session")

    def process_om_session(self, user_id: str, session: str):
        rgb_file, depth_file = self.dataset_helper.get_video_files(user_id, session)
        speech_labels = self.dataset_helper.load_speech_labels(user_id, session)
        # Log loaded data shapes
        rospy.loginfo(f"    RGB file: {rgb_file}")
        rospy.loginfo(f"    Depth file: {depth_file}")
        if speech_labels is not None:
            rospy.loginfo(f"    Speech labels: {len(speech_labels)} lines")
        # TODO: Add OM-specific evaluation logic here

    def initialize_model(self) -> Optional[AutoModelForCausalLM]:
        """Initialize the Moondream 2 model with error handling."""
        try:
            rospy.loginfo("\nInitializing Moondream 2 model...")
            if torch.cuda.is_available():
                rospy.loginfo(f"GPU detected: {torch.cuda.get_device_name(0)}")
                device = "cuda"
            else:
                rospy.loginfo("No GPU detected, using CPU")
                device = "cpu"
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                revision=self.model_revision,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                device_map={"": device} if device == "cuda" else None,
            )
            if device == "cpu":
                model = model.to(device)
            model.eval()
            rospy.loginfo("✓ Model initialized successfully")
            return model
        except Exception as e:
            rospy.logerr(f"\nError initializing model: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_gaze_from_frame(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        from PIL import Image as PILImage
        pil_image = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detection_result = self.model.detect(pil_image, "face")
        if isinstance(detection_result, dict) and "objects" in detection_result:
            faces = detection_result["objects"]
        elif isinstance(detection_result, list):
            faces = detection_result
        else:
            return None
        if not faces:
            return None
        face = faces[0]  # Should only be one face
        face_center = (
            float(face["x_min"] + face["x_max"]) / 2,
            float(face["y_min"] + face["y_max"]) / 2,
        )
        try:
            gaze_result = self.model.detect_gaze(pil_image, face_center)
            if isinstance(gaze_result, dict) and "gaze" in gaze_result:
                gaze = gaze_result["gaze"]
            else:
                gaze = gaze_result
            if gaze and isinstance(gaze, dict) and "x" in gaze and "y" in gaze:
                return float(gaze["x"]), float(gaze["y"])
        except Exception as e:
            rospy.logwarn(f"Error detecting gaze: {e}")
        return None

    def get_depth_at_pixel(self, depth_frame: Optional[np.ndarray], u_norm: float, v_norm: float, filtering: bool = True, filter_width: int = 3) -> Optional[float]:
        # Extract the depth value from the depth frame at normalized coordinates (u_norm, v_norm)
        if depth_frame is None:
            return None
        h, w = depth_frame.shape[:2]
        x = int(u_norm * w)
        y = int(v_norm * h)
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

if __name__ == "__main__":
    try:
        rospy.init_node('gaze_detection_evaluator', anonymous=True)
        evaluator = GazeDetectionEvaluator()
        evaluator.evaluate_dataset()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
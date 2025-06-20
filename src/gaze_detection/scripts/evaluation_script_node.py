#!/usr/bin/env python3

import os
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
        return [d for d in os.listdir(self.dataset_path) if d.startswith('ID_') and os.path.isdir(os.path.join(self.dataset_path, d))]

    def get_session_path(self, user_id: str, session: str) -> str:
        return os.path.join(self.dataset_path, user_id, session)

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
        self.dataset_dir = rospy.get_param('~dataset_dir', '/path/to/dataset')
        self.results_dir = rospy.get_param('~results_dir', '/path/to/results')
        self.model_id = rospy.get_param('~model_id', "vikhyatk/moondream2")
        self.model_revision = rospy.get_param('~model_revision', "2025-01-09")
        self.bridge = CvBridge()
        self.processing_times = []
        self.processed_count = 0

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
        session_order = ['ET_center', 'MT']
        total_processed = 0
        for user_id in id_list:
            rospy.loginfo(f"Processing {user_id}")
            for session in session_order:
                session_path = self.dataset_helper.get_session_path(user_id, session)
                if not os.path.exists(session_path):
                    rospy.logwarn(f"  Session {session} not found for {user_id}")
                    continue
                rospy.loginfo(f"  Processing session {session}")
                # Load data
                rgb_file, depth_file = self.dataset_helper.get_video_files(user_id, session)
                mouse_events = self.dataset_helper.load_mouse_events(user_id, session)
                gaze_labels = self.dataset_helper.load_gaze_labels(user_id, session)
                speech_labels = self.dataset_helper.load_speech_labels(user_id, session)
                left_arm = self.dataset_helper.load_left_arm(user_id, session) if session == 'ET_center' else None
                right_arm = self.dataset_helper.load_right_arm(user_id, session) if session == 'ET_center' else None
                # Example: Log loaded data shapes
                rospy.loginfo(f"    RGB file: {rgb_file}")
                rospy.loginfo(f"    Depth file: {depth_file}")
                if mouse_events is not None:
                    rospy.loginfo(f"    Mouse events: {mouse_events.shape}")
                if gaze_labels is not None:
                    rospy.loginfo(f"    Gaze labels: {gaze_labels.shape}")
                if speech_labels is not None:
                    rospy.loginfo(f"    Speech labels: {len(speech_labels)} lines")
                if left_arm is not None:
                    rospy.loginfo(f"    Left arm: {left_arm.shape}")
                if right_arm is not None:
                    rospy.loginfo(f"    Right arm: {right_arm.shape}")
                # TODO: Add your evaluation logic here
                # For now, just count processed sessions
                total_processed += 1
        rospy.loginfo(f"Finished evaluating {total_processed} sessions.")

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

if __name__ == "__main__":
    try:
        rospy.init_node('gaze_detection_evaluator', anonymous=True)
        evaluator = GazeDetectionEvaluator()
        evaluator.evaluate_dataset()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
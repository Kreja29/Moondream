#!/usr/bin/env python3

import os
import sys
import rospy
import cv2
import numpy as np
from typing import List, Dict, Tuple, Any
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import traceback

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

        rospy.loginfo("GazeDetectionEvaluator initialized")

    def evaluate_dataset(self):
        # TODO: Implement dataset evaluation logic here
        rospy.loginfo(f"Evaluating dataset in {self.dataset_dir}")
        # Example: iterate over images in dataset_dir
        for filename in os.listdir(self.dataset_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_path = os.path.join(self.dataset_dir, filename)
                try:
                    # Read image
                    frame = cv2.imread(image_path)
                    if frame is None:
                        rospy.logwarn(f"Failed to read image: {image_path}")
                        continue

                    # TODO: Run processing/evaluation here
                    # result = self.process_frame(frame)

                    # TODO: Save or log results as needed

                    self.processed_count += 1
                except Exception as e:
                    rospy.logerr(f"Error processing {filename}: {e}")
                    traceback.print_exc()

        rospy.loginfo(f"Finished evaluating {self.processed_count} images.")

    # def process_frame(self, frame: np.ndarray) -> Any:
    #     # TODO: Implement frame processing/evaluation logic
    #     pass

    def initialize_model(self) -> Optional[AutoModelForCausalLM]:
        """Initialize the Moondream 2 model with error handling."""
        try:
            rospy.loginfo("\nInitializing Moondream 2 model...")
            import torch
            from transformers import AutoModelForCausalLM
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
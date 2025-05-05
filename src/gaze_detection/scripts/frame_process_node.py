#!/usr/bin/env python3

import os
import sys
import rospy
import numpy as np
import cv2
from PIL import Image as PILImage
from transformers import AutoModelForCausalLM
import torch
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from typing import List, Dict, Optional, Tuple, Any
import traceback

class GazeDetectionProcessor:
    def __init__(self):
        # Get ROS parameters
        self.model_id = rospy.get_param('~model_id', "vikhyatk/moondream2")
        self.model_revision = rospy.get_param('~model_revision', "2025-01-09")
        
        # Processing stats
        self.processing_times = []
        self.processed_count = 0
        
        # Initialize model
        self.model = self.initialize_model()
        if self.model is None:
            rospy.logerr("Failed to initialize Moondream 2 model. Exiting.")
            sys.exit(1)

        # Subscribe to point cloud topic
        self.pointcloud_sub = rospy.Subscriber(
            "/kinect/depth_registered/points",
            PointCloud2,
            self.pointcloud_callback,
            queue_size=1
        )
        
        rospy.loginfo("GazeDetectionProcessor initialized")

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
            traceback.print_exc()
            return None

    def pointcloud_to_image(self, cloud_msg: PointCloud2) -> np.ndarray:
        """Convert PointCloud2 message to 2D image"""
        # Convert point cloud to numpy array
        points = np.array(list(pc2.read_points(cloud_msg, skip_nans=True)))
        
        if len(points) == 0:
            return None

        # Extract x, y, z coordinates
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        # Project 3D points to 2D plane (simple orthographic projection)
        height = int(abs(y.max() - y.min()) * 100)  # Scale factor of 100
        width = int(abs(x.max() - x.min()) * 100)   # Scale factor of 100
        
        # Ensure minimum dimensions
        height = max(height, 480)
        width = max(width, 640)

        # Create depth image
        image = np.zeros((height, width), dtype=np.uint8)
        
        # Normalize and scale depth values to 0-255
        depth_normalized = ((z - z.min()) / (z.max() - z.min()) * 255).astype(np.uint8)
        
        # Map points to image pixels
        x_img = ((x - x.min()) / (x.max() - x.min()) * (width - 1)).astype(int)
        y_img = ((y - y.min()) / (y.max() - y.min()) * (height - 1)).astype(int)
        
        # Assign depth values to image
        image[y_img, x_img] = depth_normalized

        # Convert to 3-channel image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        return image

    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """Process a single frame to detect faces and gaze coordinates"""
        try:
            # Convert frame for model
            pil_image = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Detect faces
            detection_result = self.model.detect(pil_image, "face")

            # Handle different possible return formats
            if isinstance(detection_result, dict) and "objects" in detection_result:
                faces = detection_result["objects"]
            elif isinstance(detection_result, list):
                faces = detection_result
            else:
                return []

            # Process each face
            results = []
            for face in faces:
                if not all(k in face for k in ["x_min", "y_min", "x_max", "y_max"]):
                    continue

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
                        results.append({
                            'face_coords': {
                                'x': face_center[0],
                                'y': face_center[1]
                            },
                            'gaze_coords': {
                                'x': float(gaze["x"]),
                                'y': float(gaze["y"])
                            }
                        })
                except Exception as e:
                    rospy.logwarn(f"Error detecting gaze: {e}")
                    continue

            return results
            
        except Exception as e:
            rospy.logerr(f"Error processing frame: {e}")
            traceback.print_exc()
            return []

    def pointcloud_callback(self, cloud_msg: PointCloud2):
        """Process incoming point cloud data"""
        try:
            # Convert point cloud to image
            frame = self.pointcloud_to_image(cloud_msg)
            if frame is None:
                rospy.logwarn("Failed to convert point cloud to image")
                return

            # Process the frame
            results = self.process_frame(frame)
            
            # Log results
            if results:
                rospy.loginfo(f"Detected {len(results)} faces with gaze coordinates")
                for i, result in enumerate(results):
                    rospy.loginfo(f"Face {i+1}: {result}")
            
        except Exception as e:
            rospy.logerr(f"Error in pointcloud callback: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    try:
        rospy.init_node('gaze_detection_processor', anonymous=True)
        processor = GazeDetectionProcessor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
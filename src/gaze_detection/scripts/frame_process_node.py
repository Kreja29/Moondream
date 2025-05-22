#!/usr/bin/env python3

import os
import sys
import rospy
import ros_numpy
import numpy as np
import cv2
import open3d
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped
from PIL import Image as PILImage
from transformers import AutoModelForCausalLM
import torch
from sensor_msgs.msg import PointCloud2, Image
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
from typing import List, Dict, Optional, Tuple, Any
import traceback
from message_filters import ApproximateTimeSynchronizer, Subscriber

class GazeDetectionProcessor:
    def __init__(self):
        # Get ROS parameters
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

        # Message filters for synchronized callback (Subscription)
        self.rgb_sub = Subscriber("/snapshot/image", Image)
        self.pc_sub = Subscriber("/snapshot/points", PointCloud2)

        self.ts = ApproximateTimeSynchronizer([self.rgb_sub, self.pc_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.synced_callback)
        
        rospy.loginfo("GazeDetectionProcessor initialized")

    def initialize_model(self) -> Optional[AutoModelForCausalLM]:
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
        
    def synced_callback(self, img_msg, pc_msg):
        try:
            self.rgb_to_pc_map = self.generate_rbg_to_pc_map(
            fx_rgb=525.0, 
            fy_rgb=525.0,
            cx_rgb=319.5,
            cy_rgb=239.5,
            organized_pc=ros_numpy.point_cloud2.pointcloud2_to_array(pc_msg).reshape((pc_msg.height, pc_msg.width, 3)) 
            )
            
            # Convert image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')

            # Process image to get 2D pixel coordinates
            gaze_results = self.process_frame(cv_image)

            # Print results
            for result in gaze_results:
                rospy.loginfo(f"Face coordinates: {result['face_coords']}")
                rospy.loginfo(f"Gaze coordinates: {result['gaze_coords']}")

        except Exception as e:
            rospy.logerr(f"Error in synced_callback: {e}")
            traceback.print_exc()

    def generate_rgb_to_pc_map(fx_rgb: float, fy_rgb: float, cx_rgb: float, cy_rgb: float, organized_pc) -> np.ndarray:

        rgb_to_pc_map = np.zeros((640, 480, 2), dtype=np.float32)

        for v in range(480):
            for u in range(640):

                point = organized_pc[v, u]
                if not (np.isnan(point[0]) or np.isnan(point[1]) or np.isnan(point[2])):
                    # Transform to RGB camera frame
                    pt = PointStamped()
                    pt.header.frame_id = "camera_depth_optical_frame"
                    # pt.header.stamp = rospy.Time(0)
                    pt.point.x = point[0]
                    pt.point.y = point[1]
                    pt.point.z = point[2]

                    pt_transformed = tf2_geometry_msgs.do_transform_point(pt, tf2_ros.Buffer().lookup_transform("camera_depth_optical_frame", "camera_rgb_optical_frame", rospy.Time(0), rospy.Duration(1.0)))

                    x_rgb, y_rgb, z_rgb = pt_transformed.point.x, pt_transformed.point.y, pt_transformed.point.z

                    u_rgb = int((fx_rgb * x_rgb / z_rgb) + cx_rgb)
                    v_rgb = int((fy_rgb * y_rgb / z_rgb) + cy_rgb)
                    
                    if 0 <= u_rgb < 640 and 0 <= v_rgb < 480:
                        # Store the mapping
                        rgb_to_pc_map[u_rgb, v_rgb] = (u, v)
        return rgb_to_pc_map
    
    def get_corresponding_point(u_img, v_img, rgb_to_pc_map, organized_pc) -> Optional[np.ndarray]:
        if (u_img, v_img) in rgb_to_pc_map:
            u_pc, v_pc = rgb_to_pc_map[u_img, v_img]
            point = organized_pc[v_pc, u_pc]
            if not (np.isnan(point[0]) or np.isnan(point[1]) or np.isnan(point[2])):
                return point

    def get_face_and_gaze_coordinates(self, image: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        TODO
        """
        h, w, _ = image.shape
        return (w // 2, h // 2), (50, 50)
    

    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """Process a single frame to detect faces and gaze coordinates"""
        try:
            image_height, image_width = frame.shape[:2]

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
                                'x': int(face_center[0] * image_width),
                                'y': int(face_center[1] * image_height)
                            },
                            'gaze_coords': {
                                'x': int(float(gaze["x"]) * image_width),
                                'y': int(float(gaze["y"]) * image_height)
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

if __name__ == "__main__":
    try:
        rospy.init_node('gaze_detection_processor', anonymous=True)
        processor = GazeDetectionProcessor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
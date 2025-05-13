#!/usr/bin/env python3

import os
import sys
import rospy
import ros_numpy
import numpy as np
import cv2
import open3d
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
        self.rgb_sub = Subscriber("/camera/rgb/image_color", Image)
        self.pc_sub = Subscriber("/camera/depth_registered/points", PointCloud2)

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
            # Convert image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')

            # Process image to get 2D pixel coordinates
            (x1, y1), (x2, y2) = self.get_face_and_gaze_coordinates(cv_image)

            # Get corresponding 3D points
            point1 = self.get_point_at_pixel(pc_msg, x1, y1)
            point2 = self.get_point_at_pixel(pc_msg, x2, y2)

            rospy.loginfo(f"Face and Gaze points in image: ({x1}, {y1}), ({x2}, {y2})")
            rospy.loginfo(f"Corresponding 3D points: {point1}, {point2}")

            # Convert entire pointcloud to Open3D format
            # open3d_cloud = self.convert_to_open3d(pc_msg)

        except Exception as e:
            rospy.logerr(f"Error in synced_callback: {e}")
            traceback.print_exc()

    def get_face_and_gaze_coordinates(self, image: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        TODO
        """
        h, w, _ = image.shape
        return (w // 2, h // 2), (50, 50)
    
    def get_point_at_pixel(self, cloud_msg: PointCloud2, x: int, y: int) -> Optional[np.ndarray]:
        """
        Retrieves the 3D point (x, y, z) corresponding to the image pixel (x, y)
        from an organized PointCloud2 message.
        """
        width, height = cloud_msg.width, cloud_msg.height
        if not (0 <= x < width and 0 <= y < height):
            return None

        index = y * width + x
        gen = pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=False)
        for i, point in enumerate(gen):
            if i == index:
                return np.array(point) if not any(np.isnan(point)) else None
        return None

    def convert_to_open3d(rospc, remove_nans=False):
        """ covert ros point cloud to open3d point cloud
        Args: 
            rospc (sensor.msg.PointCloud2): ros point cloud message
            remove_nans (bool): if true, ignore the NaN points
        Returns: 
            o3dpc (open3d.geometry.PointCloud): open3d point cloud
        """
        field_names = [field.name for field in rospc.fields]
        is_rgb = 'rgb' in field_names
        cloud_array = ros_numpy.point_cloud2.pointcloud2_to_array(rospc).ravel()
        if remove_nans:
            mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
            cloud_array = cloud_array[mask]
        if is_rgb:
            cloud_npy = np.zeros(cloud_array.shape + (4,), dtype=float)
        else: 
            cloud_npy = np.zeros(cloud_array.shape + (3,), dtype=float)
        
        cloud_npy[...,0] = cloud_array['x']
        cloud_npy[...,1] = cloud_array['y']
        cloud_npy[...,2] = cloud_array['z']
        o3dpc = open3d.geometry.PointCloud()

        if len(np.shape(cloud_npy)) == 3:
            cloud_npy = np.reshape(cloud_npy[:, :, :3], [-1, 3], 'F')
        o3dpc.points = open3d.utility.Vector3dVector(cloud_npy[:, :3])

        if is_rgb:
            rgb_npy = cloud_array['rgb']
            rgb_npy.dtype = np.uint32
            r = np.asarray((rgb_npy >> 16) & 255, dtype=np.uint8)
            g = np.asarray((rgb_npy >> 8) & 255, dtype=np.uint8)
            b = np.asarray(rgb_npy & 255, dtype=np.uint8)
            rgb_npy = np.asarray([r, g, b])
            rgb_npy = rgb_npy.astype(float)/255
            rgb_npy = np.swapaxes(rgb_npy, 0, 1)
            o3dpc.colors = open3d.utility.Vector3dVector(rgb_npy)
        return o3dpc

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

if __name__ == "__main__":
    try:
        rospy.init_node('gaze_detection_processor', anonymous=True)
        processor = GazeDetectionProcessor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
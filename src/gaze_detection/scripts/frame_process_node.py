#!/usr/bin/env python3

import os
import sys
import rospy
import ros_numpy
import numpy as np
import cv2
import open3d
import tf2_ros
from geometry_msgs.msg import PointStamped, TransformStamped, Point
import tf.transformations as tft
from PIL import Image as PILImage
from transformers import AutoModelForCausalLM
import torch
from sensor_msgs.msg import PointCloud2, Image
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
from typing import List, Dict, Optional, Tuple, Any
import traceback
from visualization_msgs.msg import Marker
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

        self.marker_pub = rospy.Publisher("visualization_marker", Marker, queue_size=10)

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
        

    def publish_arrow_marker(self, start_point: Tuple[float, float, float], end_point: Tuple[float, float, float], marker_id: int = 0):
        marker = Marker()
        marker.header.frame_id = "camera_depth_optical_frame"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "gaze_arrow"
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        # Set the arrow's start and end points
        marker.points = [
            PointStamped(point=geometry_msgs.msg.Point(*start_point)).point,
            PointStamped(point=geometry_msgs.msg.Point(*end_point)).point,
        ]

        # Arrow scale: shaft diameter, head diameter, head length
        marker.scale.x = 0.01
        marker.scale.y = 0.02
        marker.scale.z = 0.02

        # Arrow color: red with some transparency
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        self.marker_pub.publish(marker)

        
    def synced_callback(self, img_msg, pc_msg):
        try:
            #self.organized_pc=ros_numpy.point_cloud2.pointcloud2_to_array(pc_msg).reshape((pc_msg.height, pc_msg.width, 3)) 

            pc_array = ros_numpy.point_cloud2.pointcloud2_to_array(pc_msg)
            pc_array = pc_array.reshape((pc_msg.height, pc_msg.width))

            self.organized_pc = np.stack((pc_array['x'], pc_array['y'], pc_array['z']), axis=-1)


            self.rgb_to_pc_map = self.generate_rgb_to_pc_map(
            fx_rgb=525.0, 
            fy_rgb=525.0,
            cx_rgb=319.5,
            cy_rgb=239.5,
            organized_pc=self.organized_pc
            )
            
            # Convert image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')

            # Process image to get 2D pixel coordinates
            gaze_results = self.process_frame(cv_image)

            rospy.loginfo("Face coordinates and Gaze coordinates detected")

            # Print results
            for result in gaze_results:
                u_f = result['face_coords']['x']
                v_f = result['face_coords']['y']
                u_g = result['gaze_coords']['x']
                v_g = result['gaze_coords']['y']
                rospy.loginfo(f"Face center: ({u_f}, {v_f})")
                rospy.loginfo(f"Gaze coordinates: ({u_g}, {v_g})")
                # Get corresponding 3D point
                point_3d_f = self.get_corresponding_point(u_f, v_f)
                if point_3d_f:
                    rospy.loginfo(f"Corresponding 3D point: ({point_3d_f.x}, {point_3d_f.y}, {point_3d_f.z})")
                point_3d_g = self.get_corresponding_point(u_g, v_g)
                if point_3d_g:
                    rospy.loginfo(f"Corresponding 3D point for gaze: ({point_3d_g.x}, {point_3d_g.y}, {point_3d_g.z})")


        except Exception as e:
            rospy.logerr(f"Error in synced_callback: {e}")
            traceback.print_exc()

    def generate_rgb_to_pc_map(self, fx_rgb: float, fy_rgb: float, cx_rgb: float, cy_rgb: float, organized_pc) -> np.ndarray:

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

                    #pt_transformed = tf2_geometry_msgs.do_transform_point(pt, tf2_ros.Buffer().lookup_transform("camera_depth_optical_frame", "camera_rgb_optical_frame", rospy.Time(0), rospy.Duration(1.0)))

                    translation = [-0.025, 0.0, 0.0]
                    rotation = [0.0, 0.0, 0.0, 1.0]

                    # Build the transformation matrix
                    transform_matrix = tft.concatenate_matrices(
                        tft.translation_matrix(translation),
                        tft.quaternion_matrix(rotation)
                    )
                    # Apply the transformation
                    point_np = np.array([pt.point.x, pt.point.y, pt.point.z, 1.0])
                    transformed_point = np.dot(transform_matrix, point_np)


                    x_rgb, y_rgb, z_rgb = transformed_point[0], transformed_point[1], transformed_point[2]

                    u_rgb = int((fx_rgb * x_rgb / z_rgb) + cx_rgb)
                    v_rgb = int((fy_rgb * y_rgb / z_rgb) + cy_rgb)
                    
                    if 0 <= u_rgb < 640 and 0 <= v_rgb < 480:
                        # Store the mapping
                        rgb_to_pc_map[u_rgb, v_rgb] = (u, v)
        return rgb_to_pc_map

    def get_corresponding_point(self, u_img: int, v_img: int, sigma: int = 3) -> Optional[Point]:
        """
        Look up the 3D point corresponding to (u_img, v_img) in the RGB image using a sigma neighborhood.
        Applies a median filter over the corresponding points from the point cloud.
        """
        if self.rgb_to_pc_map is None or self.organized_pc is None:
            return None

        height, width = self.rgb_to_pc_map.shape[:2]
        pc_height, pc_width = self.organized_pc.shape[:2]
        points = []

        for du in range(-sigma, sigma + 1):
            for dv in range(-sigma, sigma + 1):
                u = u_img + du
                v = v_img + dv

                if 0 <= u < width and 0 <= v < height:
                    u_pc, v_pc = self.rgb_to_pc_map[u, v]
                    u_pc, v_pc = int(u_pc), int(v_pc)

                    if (u_pc != 0 or v_pc != 0) and 0 <= u_pc < pc_width and 0 <= v_pc < pc_height:
                        point = self.organized_pc[v_pc, u_pc]
                        if not np.isnan(point[0]) and not np.isnan(point[1]) and not np.isnan(point[2]):
                            points.append([point[0], point[1], point[2]])

        if not points:
            return None

        median = np.median(np.array(points), axis=0)
        return Point(x=float(median[0]), y=float(median[1]), z=float(median[2]))


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
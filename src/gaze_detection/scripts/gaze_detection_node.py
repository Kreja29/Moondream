#!/usr/bin/env python3
"""
Gaze Detection ROS Node using Moondream 2
-----------------------------------------
ROS node for face and gaze detection in video streams
"""

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from transformers import AutoModelForCausalLM
import os
from typing import List, Dict, Optional
import time
from contextlib import contextmanager


class GazeDetectionNode:
    def __init__(self):
        rospy.init_node('gaze_detection_node')
        
        # Get parameters
        self.input_topic = rospy.get_param('~input_topic', '/camera/image_raw')
        self.output_topic = rospy.get_param('~output_topic', '/gaze_detection/image')
        self.visualization_enabled = rospy.get_param('~visualization', True)
        self.model_id = rospy.get_param('~model_id', 'vikhyatk/moondream2')
        self.model_revision = rospy.get_param('~model_revision', '2025-01-09')
        
        # Initialize the bridge between ROS and OpenCV
        self.bridge = CvBridge()
        
        # Initialize publishers and subscribers
        self.image_sub = rospy.Subscriber(self.input_topic, Image, self.image_callback)
        self.image_pub = rospy.Publisher(self.output_topic, Image, queue_size=1)
        
        # Stats variables
        self.frame_count = 0
        self.processing_times = []
        self.last_stats_time = time.time()
        
        # Initialize the model
        self.model = self.initialize_model()
        if self.model is None:
            rospy.logerr("Failed to initialize Moondream 2 model. Shutting down.")
            rospy.signal_shutdown("Model initialization failed")
            return
        
        rospy.loginfo(f"Gaze Detection Node initialized. Subscribing to {self.input_topic}")
        rospy.loginfo(f"Publishing detected results to {self.output_topic}")

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

            rospy.loginfo(f"Loading model '{self.model_id}' (revision: {self.model_revision}) from HuggingFace...")
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
            return None

    def fig2rgb_array(self, fig: plt.Figure) -> np.ndarray:
        """Convert matplotlib figure to RGB array"""
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        w, h = fig.canvas.get_width_height()
        img_array = np.asarray(buf).reshape((h, w, 4))
        rgb_array = img_array[:, :, :3]  # Drop alpha channel
        return rgb_array

    def visualize_frame(self, frame: np.ndarray, faces: List[Dict], pil_image: PILImage) -> np.ndarray:
        """Visualize a single frame using matplotlib"""
        try:
            # Create figure without margins
            fig = plt.figure(figsize=(frame.shape[1] / 100, frame.shape[0] / 100), dpi=100)
            ax = fig.add_axes([0, 0, 1, 1])

            # Display frame
            ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Sort faces by x_min coordinate for stable colors
            faces = sorted(faces, key=lambda f: (f["y_min"], f["x_min"]))

            # Generate colors
            colors = plt.cm.rainbow(np.linspace(0, 1, max(1, len(faces))))

            # Process each face
            for face, color in zip(faces, colors):
                try:
                    # Calculate face box coordinates
                    x_min = int(float(face["x_min"]) * frame.shape[1])
                    y_min = int(float(face["y_min"]) * frame.shape[0])
                    width = int(float(face["x_max"] - face["x_min"]) * frame.shape[1])
                    height = int(float(face["y_max"] - face["y_min"]) * frame.shape[0])

                    # Draw face rectangle
                    rect = plt.Rectangle(
                        (x_min, y_min), width, height, fill=False, color=color, linewidth=2
                    )
                    ax.add_patch(rect)

                    # Calculate face center
                    face_center = (
                        float(face["x_min"] + face["x_max"]) / 2,
                        float(face["y_min"] + face["y_max"]) / 2,
                    )

                    # Try to detect gaze
                    try:
                        gaze_result = self.model.detect_gaze(pil_image, face_center)
                        if isinstance(gaze_result, dict) and "gaze" in gaze_result:
                            gaze = gaze_result["gaze"]
                        else:
                            gaze = gaze_result
                    except Exception as e:
                        rospy.logwarn(f"Error detecting gaze: {e}")
                        continue

                    if (
                        gaze is not None
                        and isinstance(gaze, dict)
                        and "x" in gaze
                        and "y" in gaze
                    ):
                        gaze_x = int(float(gaze["x"]) * frame.shape[1])
                        gaze_y = int(float(gaze["y"]) * frame.shape[0])
                        face_center_x = x_min + width // 2
                        face_center_y = y_min + height // 2

                        # Draw gaze line with gradient effect
                        points = 50
                        alphas = np.linspace(0.8, 0, points)

                        # Calculate points along the line
                        x_points = np.linspace(face_center_x, gaze_x, points)
                        y_points = np.linspace(face_center_y, gaze_y, points)

                        # Draw gradient line segments
                        for i in range(points - 1):
                            ax.plot(
                                [x_points[i], x_points[i + 1]],
                                [y_points[i], y_points[i + 1]],
                                color=color,
                                alpha=alphas[i],
                                linewidth=4,
                            )

                        # Draw gaze point
                        ax.scatter(gaze_x, gaze_y, color=color, s=100, zorder=5)
                        ax.scatter(gaze_x, gaze_y, color="white", s=50, zorder=6)

                except Exception as e:
                    rospy.logwarn(f"Error processing face: {e}")
                    continue

            # Configure axes
            ax.set_xlim(0, frame.shape[1])
            ax.set_ylim(frame.shape[0], 0)
            ax.axis("off")

            # Convert matplotlib figure to image
            frame_rgb = self.fig2rgb_array(fig)

            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Clean up
            plt.close(fig)

            return frame_bgr

        except Exception as e:
            rospy.logwarn(f"Error in visualize_frame: {e}")
            plt.close("all")
            return frame

    def process_frame(self, frame):
        """Process a single frame to detect faces and gaze"""
        start_time = time.time()
        
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
                rospy.logwarn(f"Unexpected detection result format: {type(detection_result)}")
                faces = []

            # Ensure each face has the required coordinates
            faces = [
                face
                for face in faces
                if all(k in face for k in ["x_min", "y_min", "x_max", "y_max"])
            ]

            if not faces:
                processed_frame = frame
            else:
                # Visualize frame with matplotlib if visualization is enabled
                if self.visualization_enabled:
                    processed_frame = self.visualize_frame(frame, faces, pil_image)
                else:
                    processed_frame = frame

            # Force matplotlib to clean up
            plt.close("all")
            
            # Update processing stats
            self.frame_count += 1
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Log stats every 10 seconds
            if time.time() - self.last_stats_time > 10 and self.processing_times:
                avg_time = sum(self.processing_times) / len(self.processing_times)
                fps = 1.0 / avg_time if avg_time > 0 else 0
                rospy.loginfo(f"Processing stats: {fps:.2f} FPS (avg: {avg_time*1000:.1f}ms/frame)")
                self.processing_times = []
                self.last_stats_time = time.time()
            
            return processed_frame
            
        except Exception as e:
            rospy.logerr(f"Error processing frame: {e}")
            plt.close("all")  # Clean up even on error
            return frame

    def image_callback(self, data):
        """Callback function for image messages"""
        try:
            # Convert ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            
            # Process the frame
            processed_frame = self.process_frame(cv_image)
            
            # Convert processed frame back to ROS Image message
            processed_msg = self.bridge.cv2_to_imgmsg(processed_frame, "bgr8")
            processed_msg.header = data.header  # Preserve the original header
            
            # Publish the processed image
            self.image_pub.publish(processed_msg)
            
        except CvBridgeError as e:
            rospy.logerr("CV Bridge error: %s", e)
        except Exception as e:
            rospy.logerr("Error in image callback: %s", e)

    def run(self):
        """Main loop"""
        rospy.spin()


if __name__ == "__main__":
    try:
        node = GazeDetectionNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        # Clean up matplotlib resources
        plt.close("all")
#!/usr/bin/env python3
"""
Moondream 2 Gaze Detection Image Processing Script
-------------------------------------------------
Processes images from the input directory and saves processed images to output directory
"""

import os
import sys
import glob
import rospy
import numpy as np
import cv2
from PIL import Image as PILImage
from transformers import AutoModelForCausalLM
import matplotlib.pyplot as plt
import torch
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple

class GazeDetectionProcessor:
    def __init__(self):
        # Get parameters
        self.input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../input")
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../output")
        self.model_id = "vikhyatk/moondream2"
        self.model_revision = "2025-01-09"
        
        # Initialize stats variables
        self.processing_times = []
        
        # Initialize the model
        self.model = self.initialize_model()
        if self.model is None:
            rospy.logerr("Failed to initialize Moondream 2 model. Exiting.")
            sys.exit(1)
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        rospy.loginfo(f"Gaze Detection Processor initialized.")
        rospy.loginfo(f"Input directory: {self.input_dir}")
        rospy.loginfo(f"Output directory: {self.output_dir}")

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
            for i, (face, color) in enumerate(zip(faces, colors)):
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
                    
                    # Add face number
                    ax.text(x_min, y_min - 5, f"Face #{i+1}", color=color, 
                            fontsize=10, weight='bold', backgroundcolor='black')

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

    def process_image(self, image_path: str) -> Tuple[np.ndarray, Dict]:
        """Process a single image to detect faces and gaze"""
        start_time = time.time()
        results = {"faces": 0, "with_gaze": 0}
        
        try:
            # Read image
            frame = cv2.imread(image_path)
            if frame is None:
                rospy.logwarn(f"Failed to read image: {image_path}")
                return None, results
            
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

            # Update results
            results["faces"] = len(faces)

            # Count faces with gaze
            for face in faces:
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
                        
                    if (
                        gaze is not None
                        and isinstance(gaze, dict)
                        and "x" in gaze
                        and "y" in gaze
                    ):
                        results["with_gaze"] += 1
                except Exception as e:
                    rospy.logwarn(f"Error detecting gaze: {e}")

            if not faces:
                processed_frame = frame
            else:
                # Visualize frame with matplotlib
                processed_frame = self.visualize_frame(frame, faces, pil_image)

            # Force matplotlib to clean up
            plt.close("all")
            
            # Update processing stats
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            results["processing_time"] = processing_time
            
            return processed_frame, results
            
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
            plt.close("all")  # Clean up even on error
            return None, results

    def process_directory(self) -> None:
        """Process all images in the input directory"""
        # Get all image files in the input directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(self.input_dir, f'*{ext}')))
            image_files.extend(glob.glob(os.path.join(self.input_dir, f'*{ext.upper()}')))
        
        if not image_files:
            rospy.logwarn(f"No image files found in {self.input_dir}")
            return
        
        rospy.loginfo(f"Found {len(image_files)} image files to process")
        
        # Process each image
        for i, image_path in enumerate(image_files):
            rospy.loginfo(f"Processing image {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
            
            # Process image
            processed_frame, results = self.process_image(image_path)
            
            if processed_frame is not None:
                # Generate output filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                basename = os.path.splitext(os.path.basename(image_path))[0]
                output_filename = f"{basename}_processed_{timestamp}.jpg"
                output_path = os.path.join(self.output_dir, output_filename)
                
                # Save processed image
                cv2.imwrite(output_path, processed_frame)
                
                rospy.loginfo(f"Saved processed image to {output_filename}")
                rospy.loginfo(f"Stats: {results['faces']} faces detected, {results['with_gaze']} with gaze")
                rospy.loginfo(f"Processing time: {results['processing_time']:.2f} seconds")
            else:
                rospy.logwarn(f"Failed to process image: {os.path.basename(image_path)}")
        
        # Print summary
        if self.processing_times:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            rospy.loginfo(f"Processing complete. Processed {len(image_files)} images.")
            rospy.loginfo(f"Average processing time: {avg_time:.2f} seconds per image")


if __name__ == "__main__":
    try:
        # Initialize ROS node
        rospy.init_node('gaze_detection_processor', anonymous=True)
        
        # Create and run processor
        processor = GazeDetectionProcessor()
        processor.process_directory()
        
    except rospy.ROSInterruptException:
        pass
    finally:
        # Clean up matplotlib resources
        plt.close("all") 
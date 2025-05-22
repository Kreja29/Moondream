#!/usr/bin/env python

import rospy
import message_filters
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import numpy as np

class GazeTo3DMapper:
    def __init__(self):
        rospy.init_node('gaze_to_3d_mapper')

        self.bridge = CvBridge()

        # Subscribers
        image_sub = message_filters.Subscriber('/camera/rgb/image_raw', Image)
        pc_sub = message_filters.Subscriber('/camera/depth_registered/points', PointCloud2)

        ats = message_filters.ApproximateTimeSynchronizer([image_sub, pc_sub], queue_size=5, slop=0.1)
        ats.registerCallback(self.callback)

        rospy.loginfo("Gaze to 3D Mapper initialized.")
        rospy.spin()

    def callback(self, img_msg, pc_msg):
        # Convert image
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        height, width, _ = cv_image.shape

        # Dummy AI model output (replace with your model)
        face_pixels = [(200, 300), (150, 150)]
        gaze_pixels = [(250, 350), (100, 50)]

        # Combine all pixels to query
        all_pixels = face_pixels + gaze_pixels

        # Point cloud is organized => use row-major index
        points = []
        for (u, v) in all_pixels:
            index = v * pc_msg.width + u  # row-major index
            gen = pc2.read_points(pc_msg, field_names=('x', 'y', 'z'), skip_nans=False)
            for i, p in enumerate(gen):
                if i == index:
                    if any(np.isnan(val) for val in p):
                        rospy.logwarn(f"NaN point at pixel ({u},{v})")
                        points.append(None)
                    else:
                        points.append(p)
                    break

        # Print results
        labels = ['face1', 'face2', 'gaze1', 'gaze2']
        for label, p in zip(labels, points):
            if p is not None:
                rospy.loginfo(f"{label} 3D point: {p}")
            else:
                rospy.loginfo(f"{label} 3D point: Invalid (NaN or out of range)")

if __name__ == '__main__':
    try:
        GazeTo3DMapper()
    except rospy.ROSInterruptException:
        pass

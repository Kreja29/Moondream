#!/usr/bin/env python3

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped

def get_static_transform(target_frame="camera_rgb_optical_frame", source_frame="camera_depth_optical_frame"):
    rospy.init_node('get_static_transform', anonymous=True)

    tf_buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_buffer)

    rospy.loginfo(f"Waiting for transform from {source_frame} to {target_frame}...")

    try:
        tf_buffer.can_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(5.0))
        transform: TransformStamped = tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            rospy.Time(0),
            rospy.Duration(5.0)
        )

        translation = transform.transform.translation
        rotation = transform.transform.rotation

        rospy.loginfo("\nTransform from %s to %s:", source_frame, target_frame)
        rospy.loginfo("Translation: [x=%.6f, y=%.6f, z=%.6f]", translation.x, translation.y, translation.z)
        rospy.loginfo("Rotation (quaternion): [x=%.6f, y=%.6f, z=%.6f, w=%.6f]",
                      rotation.x, rotation.y, rotation.z, rotation.w)

        print("\n=== COPY THIS INTO YOUR CODE ===")
        print(f"translation = [{translation.x}, {translation.y}, {translation.z}]")
        print(f"rotation = [{rotation.x}, {rotation.y}, {rotation.z}, {rotation.w}]")
        print("================================")

    except Exception as e:
        rospy.logerr(f"Error retrieving transform: {e}")

if __name__ == '__main__':
    get_static_transform()


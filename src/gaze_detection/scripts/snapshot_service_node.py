#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image, PointCloud2
import message_filters
from std_srvs.srv import Trigger, TriggerResponse
import threading

class SnapshotPublisher:
    def __init__(self):
        rospy.init_node('snapshot_service_node')

        # Publishers
        self.image_pub = rospy.Publisher('/snapshot/image', Image, queue_size=1)
        self.pc_pub = rospy.Publisher('/snapshot/points', PointCloud2, queue_size=1)

        # Subscribers
        image_sub = message_filters.Subscriber('/camera/rgb/image_color', Image)
        pc_sub = message_filters.Subscriber('/camera/depth_registered/points', PointCloud2)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [image_sub, pc_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.synced_callback)

        self.lock = threading.Lock()
        self.latest_image = None
        self.latest_pc = None

        # Service
        self.service = rospy.Service('/snapshot_trigger', Trigger, self.handle_trigger)

    def synced_callback(self, image_msg, pc_msg):
        with self.lock:
            self.latest_image = image_msg
            self.latest_pc = pc_msg

    def handle_trigger(self, req):
        with self.lock:
            if self.latest_image is not None and self.latest_pc is not None:
                self.image_pub.publish(self.latest_image)
                self.pc_pub.publish(self.latest_pc)
                rospy.loginfo("Published snapshot of image and point cloud.")
                return TriggerResponse(success=True, message="Snapshot published.")
            else:
                rospy.logwarn("No synchronized messages available yet.")
                return TriggerResponse(success=False, message="No synchronized messages yet.")

    def spin(self):
        rospy.loginfo("Snapshot service node ready. Call /snapshot_trigger to capture.")
        rospy.spin()

if __name__ == '__main__':
    try:
        node = SnapshotPublisher()
        node.spin()
    except rospy.ROSInterruptException:
        pass

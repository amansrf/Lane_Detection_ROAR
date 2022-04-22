import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy, QoSLivelinessPolicy

from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo

import cv2
from cv_bridge import CvBridge


import numpy as np

bridge = CvBridge()

path = "/home/roar/Desktop/LaneDetection/P6010001.MOV"

class RGBStreamer(Node):
    def __init__(self):
        super().__init__("rgb_streamer")

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_VOLATILE,
            liveliness=QoSLivelinessPolicy.RMW_QOS_POLICY_LIVELINESS_AUTOMATIC,
            depth=10,
        )
        self.rgb_image_pub = self.create_publisher(Image, "/rgb_image", qos_profile=qos_profile)

        self.cap = cv2.VideoCapture(path)

        timer_period = 1/60.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):

        ret, img = self.cap.read()

        if not ret:
            print("Game Over")
            rclpy.shutdown()

        #img = cv2.resize(img, (400, 400)).astype(np.uint8)
        rgb_img_msg = bridge.cv2_to_imgmsg(img, encoding="bgr8")
        rgb_img_msg.header.stamp = self.get_clock().now().to_msg()
        self.rgb_image_pub.publish(rgb_img_msg)


def main(args=None):
    rclpy.init(args=args)

    rgb_streamer = RGBStreamer()
    try:
        rclpy.spin(rgb_streamer)
    except KeyboardInterrupt:
        pass
    rgb_streamer.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
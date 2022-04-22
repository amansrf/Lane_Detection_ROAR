#import black
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
    QoSLivelinessPolicy,
)

import cv2
from cv_bridge import CvBridge
import numpy as np
from sensor_msgs.msg import Image



class RoadMaskNode(Node):
    def __init__(self):
        super().__init__("road_mask_node")
        self.bridge = CvBridge()
        self.declare_parameter("rgb_camera_topic", "rgb_image")

        self.rgb_camera_topic = (
            self.get_parameter("rgb_camera_topic").get_parameter_value().string_value
        )

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.SYSTEM_DEFAULT,
            durability=QoSDurabilityPolicy.SYSTEM_DEFAULT,
            liveliness=QoSLivelinessPolicy.SYSTEM_DEFAULT,
            depth=1,
        )
        self.img_sub = self.create_subscription(
            Image, self.rgb_camera_topic, self.img_callback, qos_profile=qos_profile
        )

        self.img_pub = self.create_publisher(Image, "/stack_img", qos_profile=qos_profile)

    def img_callback(self, msg: Image):

        # Converting ROS image message to RGB
        img1 = cv2.resize(
            self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8"), dsize=(400, 400)
        )
        img2 = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)
        # concatenate image Horizontally
        Hori = np.concatenate((img1, img2), axis=1)

        self.img_pub.publish(Hori)

        cv2.imshow("HORIZONTAL", Hori)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)

    node = RoadMaskNode()

    rclpy.spin(node)

    # Destroy the node explicitly
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

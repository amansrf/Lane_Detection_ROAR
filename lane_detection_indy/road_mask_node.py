
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
    """Filter out road from the image. Output a mask of road

    Args:
        Node (rclpy.node): Node that output mask of road
    """

    def __init__(self):
        super().__init__("road_mask_node")
        self.bridge = CvBridge()
        self.declare_parameter("rgb_camera_topic", "rgb_image")
        self.declare_parameter("debug", False)
        self.declare_parameter("ground_mask", [0, 0, 80, 255, 50, 200])
        self.declare_parameter("grass_mask", [43, 50, 20, 128, 255, 255])
        # horizon is at horizon_pct percent of image height
        self.declare_parameter("horizon_pct", 0.25)
        self.declare_parameter("morph_kernel_size", 30)

        self.morph_kernel_size = (
            self.get_parameter("morph_kernel_size").get_parameter_value().integer_value
        )

        self.horizon_pct = (
            self.get_parameter("horizon_pct").get_parameter_value().double_value
        )
        self.ground_mask = list(
            self.get_parameter("ground_mask").get_parameter_value().integer_array_value
        )
        self.grass_mask = list(
            self.get_parameter("grass_mask").get_parameter_value().integer_array_value
        )
        self.get_logger().info(f"Ground Mask: {self.ground_mask}")
        self.get_logger().info(f"Grass Mask: {self.grass_mask}")

        self.rgb_camera_topic = (
            self.get_parameter("rgb_camera_topic").get_parameter_value().string_value
        )
        self.debug = self.get_parameter("debug").get_parameter_value().bool_value

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.SYSTEM_DEFAULT,
            durability=QoSDurabilityPolicy.SYSTEM_DEFAULT,
            liveliness=QoSLivelinessPolicy.SYSTEM_DEFAULT,
            depth=1,
        )
        self.img_sub = self.create_subscription(
            Image, self.rgb_camera_topic, self.img_callback, qos_profile=qos_profile
        )

        self.mask_pub = self.create_publisher(
            Image, f"{self.rgb_camera_topic}/road_mask", qos_profile=qos_profile
        )
        self.img_sub  # prevent unused variable warning
        self.get_logger().info("Initialized")

    def img_callback(self, msg: Image):
        # Converting ROS image message to RGB
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # use color of gound and grass to mask the road
        rg_detection_mask = self.filter_road(image)

        result = self.bridge.cv2_to_imgmsg(rg_detection_mask, encoding="mono8")
        result.header.stamp = msg.header.stamp
        result.header.frame_id = msg.header.frame_id
        self.mask_pub.publish(result)

    def ground_and_grass_mask(self, image):
        hsvImg = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        ground = cv2.inRange(
            hsvImg,
            np.array(self.ground_mask[:3], dtype="uint8"),
            np.array(self.ground_mask[3:], dtype="uint8"),
        )
        grass = cv2.inRange(
            hsvImg,
            np.array(self.grass_mask[:3], dtype="uint8"),
            np.array(self.grass_mask[3:], dtype="uint8"),
        )

        black_mask = cv2.inRange(image, np.array([0, 0, 0]), np.array([50, 50, 50]))
        combined = cv2.bitwise_and(ground, cv2.bitwise_not(grass))
        combined = cv2.bitwise_and(ground, cv2.bitwise_not(black_mask))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morph_kernel_size, self.morph_kernel_size)
        )
        combined = cv2.morphologyEx(combined, cv2.MORPH_DILATE, kernel)

        return combined

    def crop_sky(self, frame: np.ndarray):
        s = frame.shape
        horizon_pixel = int(s[0] * self.horizon_pct)
        frame[:horizon_pixel, :] = [255, 255, 255]
        return frame

    def filter_road(self, frame):
        frame = self.crop_sky(frame)
        mask = self.ground_and_grass_mask(frame)
        return mask


def main(args=None):
    rclpy.init(args=args)
    node = RoadMaskNode()
    rclpy.spin(node)
    # Destroy the node explicitly
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

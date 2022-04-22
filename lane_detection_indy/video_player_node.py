#!/usr/bin/env python3
import rclpy
import rclpy.node
import sys
import cv2
import numpy as np
from pathlib import Path
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class TestVideoPlayerNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("test_video_player")
        self.declare_parameter("video_path", "NA")
        self.declare_parameter("debug", False)
        self.declare_parameter("fps", 30.0)

        self.debug = self.get_parameter("debug").get_parameter_value().bool_value
        self.fps = self.get_parameter("fps").get_parameter_value().double_value

        self.timer = self.create_timer(1 / self.fps, self.callback)
        self.video_path = (
            self.get_parameter("video_path").get_parameter_value().string_value
        )
        self.get_logger().info(f"Playing: {self.video_path}")
        self.cap = cv2.VideoCapture(self.video_path)
        self.img_pub = self.create_publisher(Image, "rgb_image", 10)
        self.bridge = CvBridge()

    def callback(self):
        ret_val, frame = self.cap.read()

        if ret_val:
            if self.debug:
                cv2.imshow("frame", frame)
                cv2.waitKey(1)
            #### SANDBOX ####
            frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
            #################
            rgb_img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            rgb_img_msg.header.frame_id = "base_link"
            self.img_pub.publish(rgb_img_msg)
        else:
            self.get_logger().info(f"Cannot open video at {self.video_path}")
            self.video_path = (
                self.get_parameter("video_path").get_parameter_value().string_value
            )
            self.cap = cv2.VideoCapture(self.video_path)


def main(args=None):
    rclpy.init()
    node = TestVideoPlayerNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy, QoSLivelinessPolicy

import cv2
from cv_bridge import CvBridge
import numpy as np
from sensor_msgs.msg import Image

import time

GROUND_MASK_LOWER = np.array([0,0,80],dtype='uint8')
GROUND_MASK_UPPER = np.array([255,50,200],dtype='uint8')
# GROUND_MASK_LOWER = np.array([0,0,40],dtype='uint8')
# GROUND_MASK_UPPER = np.array([180,30,200],dtype='uint8')
GRASS_MASK_LOWER = np.array([43,50,20],dtype='uint8')
GRASS_MASK_UPPER = np.array([128,255,255],dtype='uint8')


class LaneDetect(Node):

    def __init__(self):
        super().__init__('lane_detect')
        self.bridge = CvBridge()

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

    def img_callback(self, msg):

        tic = time.time()
        # Converting ROS image message to RGB
        image = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')

        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Smooth the image
        image = cv2.GaussianBlur(image, (5,5), 0)

        # use color of gound and grass to mask the road
        rg_detection_mask = self.filter_road(image)
        
        # Floodfill the road
        seed_value = self.get_seed_value(image)
        floodfill_mask = self.floodfill(image, seed_value)
        #floodfill_mask = cv2.morphologyEx(floodfill_mask, cv2.MORPH_CLOSE, np.ones((7,7)))
        
        # merge two masks 
        final_mask = cv2.bitwise_and(rg_detection_mask, floodfill_mask)
        opening_kernel = np.ones((5, 5))
        closing_kernel = np.ones((15, 15))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel=closing_kernel)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel=opening_kernel)
        toc = time.time()

        self.get_logger().info(f"Processing Time: {toc-tic}")
        self.mask_pub.publish(self.bridge.cv2_to_imgmsg(final_mask, encoding="mono8"))

    def get_seed_value(self, img):
        height = img.shape[0]
        width  = img.shape[1]
        # Choose bottom centre as starting point of floodfill
        seed = (height-1, width//2)
        # print(seed, height, width)
        
        # Create a box around seed and get average
        box = (200, 200)
        x = (seed[0] - box[0] - box[1], seed[0])
        y = (seed[1] - box[1], seed[1] + box[1])
        return np.mean(img[x[0]:x[1], y[0]:y[1], :], axis = (0,1))

    def floodfill(self, img, seed_value):

        img_hsv = img.copy()

        height = img.shape[0]
        width  = img.shape[1]

        # Choose bottom centre as starting point of floodfill
        seed = (height-1, width//2)
        # print(seed, height, width)
        
        # Create a box around seed and get average
   
        # TODO: Tune this as per requirement
        thre = [50, 50, 200]

        # Change the seed to calculated mean
        img_hsv[seed[0], seed[1]] = seed_value
        
        # print(img_hsv.shape, seed)

        seed = (width//2, height-1)
        mask = np.zeros((img_hsv.shape[0] + 2, img_hsv.shape[1] + 2)).astype(np.uint8)
        cv2.floodFill(
            img_hsv,
            mask,
            seedPoint=seed,
            newVal=(255, 0, 0),
            loDiff=tuple(thre),
            upDiff=tuple(thre),
            flags= cv2.FLOODFILL_FIXED_RANGE
        )

        # Resize after floodfill extension
        mask =  mask[1:-1, 1:-1] * 255
        return mask

    def ground_and_grass_mask(self, image):
        hsvImg = image
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
        frame = frame.copy()
        frame = self.crop_sky(frame)
        mask = self.ground_and_grass_mask(frame)
        return mask


def main(args=None):
    rclpy.init(args=args)

    lane_detect = LaneDetect()

    rclpy.spin(lane_detect)

    # Destroy the node explicitly
    lane_detect.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
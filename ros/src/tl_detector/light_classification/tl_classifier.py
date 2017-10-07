from styx_msgs.msg import TrafficLight
import cv2
import numpy as np

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        blurred = cv2.GaussianBlur(image, (9, 9), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        #Red
        red_lower = np.array([0,190,210], dtype=np.uint8)
        red_upper = np.array([10,255,255], dtype=np.uint8)
        red_mask = cv2.inRange(hsv, red_lower, red_upper)
        red_keypoints = self.get_blobs(red_mask)
        if len(red_keypoints) > 0:
            return TrafficLight.RED

        #Yellow
        yellow_lower = np.array([20,190,200], dtype=np.uint8)
        yellow_upper = np.array([30,255,255], dtype=np.uint8)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        yellow_keypoints = self.get_blobs(yellow_mask)
        if len(yellow_keypoints) > 0:
            return TrafficLight.YELLOW

        #Green
        green_lower = np.array([60,190,210], dtype=np.uint8)
        green_upper = np.array([80,255,255], dtype=np.uint8)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        green_keypoints = self.get_blobs(green_mask)
        if len(green_keypoints) > 0:
            return TrafficLight.GREEN

        return TrafficLight.UNKNOWN

    def get_blobs(self, mask):
        blurred = cv2.GaussianBlur(mask,(9,9),0)

        params = cv2.SimpleBlobDetector_Params()
        # Filter by Area
        params.filterByArea = False
        params.minArea = 10
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.8
        # Filter by Color
        params.filterByColor = True
        params.blobColor = 255 # white

        detector = cv2.SimpleBlobDetector_create(params)
        return detector.detect(mask)

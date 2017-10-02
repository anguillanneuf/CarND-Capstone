#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose, Point
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math
import PIL
import numpy as np

#from keras.models import load_model
#from keras.preprocessing.image import load_img, img_to_array
#from pyquaternion import Quaternion

STATE_COUNT_THRESHOLD = 3


class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        rate = rospy.Rate(2) 

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1
        
    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement: find index of cloest waypoint to pose
        n = len(self.waypoints.waypoints)
        node = [pose.position.x, pose.position.y]
        nodes = np.zeros((n,2))

        for i in range(n):
            nodes[i] = [self.waypoints.waypoints[i].pose.pose.position.x,
                        self.waypoints.waypoints[i].pose.pose.position.y]

        dist = np.sum((nodes - node)**2, axis=1)

        return np.argmin(dist)

    def project_to_image_plane(self, point_in_world):
        """Project point from 3D world coordinates to 2D camera image location
        Args:
            point_in_world (Point): 3D location of a point in the world
        Returns:
            x (int): x coordinate of target point in image
            y (int): y coordinate of target point in image
        """

        fx = self.config['camera_info']['focal_length_x']
        fy = self.config['camera_info']['focal_length_y']
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']

        # get transform between pose of camera and world frame
        trans = None
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/base_link",
                  "/world", now, rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransform("/base_link",
                  "/world", now)

        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to find camera to map transform")

        x = 0
        y = 0
        cx = image_width/2
        cy = image_height/2

        if trans != None:
            # Calculates roll (psi), pitch (theta), yaw (phi) from the rotation matrix
            euler = tf.transformations.euler_from_quaternion(rot)

            sin_yaw = math.sin(euler[2])
            cos_yaw = math.cos(euler[2])

            # Pinhole Camera Model (https://goo.gl/x2oHRu)
            # For more details, see (https://goo.gl/epdPfm)
            # Rt = Rotation * Point + translation 
            Rt = (point_in_world.x*cos_yaw - point_in_world.y*sin_yaw + trans[0], 
                point_in_world.x*sin_yaw + point_in_world.y*cos_yaw + trans[1], 
                point_in_world.z + trans[2])

            # manual tweaking
            if fx < 10:
                fx = 2574
                fy = 2744
                cx = image_width/2 - 30
                cy = image_height + 50

            # Note axis changes: car.x = img.z, car.y = img.x, car.z = img.y
            x = int(fx * (-Rt[1])/Rt[0] + cx)
            y = int(fy * (-Rt[2])/Rt[0] + cy)

        return (x, y)

    def get_light_state(self, light):
        """Determines the current color of the traffic light
        Args:
            light (TrafficLight): light to classify
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        ###TODO(denise) Replace with CV later
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        self.camera_image.encoding = "rgb8"
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        h, w, c = cv_image.shape

        x_center, y_center = self.project_to_image_plane(light.pose.pose.position)

        top = Point()
        top.x = light.pose.pose.position.x+3
        top.y = light.pose.pose.position.y-3
        bottom = Point()
        bottom.x = light.pose.pose.position.x-3 
        bottom.y = light.pose.pose.position.y+3
        x_top, y_top = self.project_to_image_plane(top)
        x_bottom, y_bottom = self.project_to_image_plane(bottom)
        print("(PRIOR) center: %d, %d top: %d, %d bottom: %d, %d" %(x_center, y_center, 
                                                            x_top, y_top,
                                                            x_bottom, y_bottom))

        if x_center < 0 or y_center < 0:
            return TrafficLight.UNKNOWN

        #crop image
        #TODO (denise) need make sure this is the correct area to crop
        cpy = cv_image.copy()

        x_top = min(600, max(x_top, 0))
        x_bottom = min(600, max(x_bottom, 0))
        y_top = min(800, max(y_top, 0))
        y_bottom = min(800, max(y_bottom, 0))

        print("(POSTERIOR) center: %d, %d top: %d, %d bottom: %d, %d" %(x_center, y_center, 
                                                            x_top, y_top,
                                                            x_bottom, y_bottom))

        #crop_img = cpy[int(y_center):int(y_top), int(x_center):int(x_top)]

        # cv2.circle(cpy,(x_center, y_center), 20, (0,255,255), 2)
        # rospy.loginfo("TL_center.x %d, TL_center.y %d", np.max(x_center, y_center))

        # write out some images
        #tm = rospy.get_rostime()
        #cv2.imwrite('traffic_lights_processed/processed_'+str(tm.secs)+'_'+str(tm.nsecs)+'.png', crop_img)

        return self.light_classifier.get_classification(cv_image)


    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']

        if self.pose and self.waypoints:
            # get the closest waypoint index for car
            car_wp = self.get_closest_waypoint(self.pose.pose)

            # get the cloesst waypoint index for next TL ahead
            min_dist = 1e2
            for i in range(len(self.lights)):
            	dist = math.hypot(self.pose.pose.position.x - self.lights[i].pose.pose.position.x,
            	                  self.pose.pose.position.y - self.lights[i].pose.pose.position.y)
                light_wp = self.get_closest_waypoint(self.lights[i].pose.pose)

                if car_wp-30 < light_wp and dist < min_dist: # manual tweak
                    min_dist = dist 
                    light_index = i
                    light = self.lights[i]

        #TODO find the closest visible traffic light (if one exists)
        if light:
            # get the closest waypoint index for upcoming stop line
            stop_line = Pose()
            stop_line.position.x = stop_line_positions[light_index][0]
            stop_line.position.y = stop_line_positions[light_index][1]
            stop_line.position.z = 0
            light_wp = self.get_closest_waypoint(stop_line)
            state = self.get_light_state(light)
            # rospy.logwarn("(tianzi) traffic light ahead at wp: %d", light_wp)
            return light_wp, state
        self.waypoints = None
        return -1, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')

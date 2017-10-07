#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math
import numpy as np
import sys

STATE_COUNT_THRESHOLD = 3
LOG = True

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
        elif self.state_count >= STATE_COUNT_THRESHOLD and self.last_state != self.state:
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

        #TODO Use tranform and rotation to calculate 2D position of light in image

        x = 0
        y = 0
        
        # focal lengths from the sim are in fact FOV in radians, convert FOV to focal lengths
        if (fx < 10):
            fx = math.tan(fx/2)*image_width*2
            fy = math.tan(fy/2)*image_height*2

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

            # Note axis changes: car.x = img.z, car.y = img.x, car.z = img.y
            x = int(fx * (-Rt[1])/Rt[0] + image_width/2)
            y = int(fy * (-Rt[2])/Rt[0] + image_height)

        return (x, y)

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #x, y = self.project_to_image_plane(light.pose.pose.position)
        #if LOG:
        #    rospy.logwarn("TL has center (%d, %d)" % (x, y))

        #TODO use light location to zoom in on traffic light in image
        #h, w, c = cv_image.shape
        #cpy = cv_image.copy()
        #cv2.circle(cpy, center = (x,y), radius = 30, color = (200, 255, 255), thickness = 5)
        #cv2.imwrite('traffic_lights_processed/processed_'+str(time.time())[:10]+'.png', cpy)
        
        # TODO: crop cv_image

        #Get classification
        result = self.light_classifier.get_classification(cv_image)
        #if LOG:
        #    rospy.logwarn("TL %s" % result)
        return result

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
            # Finds the closest waypoint index of the car
            car_wp = self.get_closest_waypoint(self.pose.pose)

            # Finds the closest waypoint index of the next stopline
            min_index_dist = sys.maxint
            for i in range(len(stop_line_positions)):
                stopline = Pose()
                stopline.position.x = stop_line_positions[i][0]
                stopline.position.y = stop_line_positions[i][1]

                stopline_wp_ = self.get_closest_waypoint(stopline)

                index_dist = abs(stopline_wp_ - car_wp)

                if index_dist < min_index_dist and car_wp < stopline_wp_:
                    min_index_dist = index_dist
                    light = self.lights[i]
                    stopline_wp = stopline_wp_

        #TODO Finds the closest visible traffic light (if one exists)
        if light:
            state = self.get_light_state(light)
            if LOG and self.last_state != self.state:
                rospy.logwarn("TL (%d), Stop line wp %d" % (state, stopline_wp))
            return stopline_wp, state
        self.waypoints = None
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
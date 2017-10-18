from styx_msgs.msg import TrafficLight
import tensorflow as tf
import cv2
import numpy as np
import rospy

class TLClassifier(object):
    def __init__(self):
        self.model = TLClassifier.load_graph()

    @staticmethod
    def load_graph():
        detection_graph = tf.Graph()
        print("Loading graph...")
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile('../../../models/tl_graph.pb', 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        print("Graph loaded!")
        return detection_graph

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        with self.model.as_default():
            with tf.Session(graph=self.model) as sess:
                # Definite input and output Tensors for detection_graph
                image_tensor = self.model.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = self.model.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = self.model.get_tensor_by_name('detection_scores:0')
                detection_classes = self.model.get_tensor_by_name('detection_classes:0')
                num_detections = self.model.get_tensor_by_name('num_detections:0')

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                  [detection_boxes, detection_scores, detection_classes, num_detections],
                  feed_dict={image_tensor: image_np_expanded})

                boxes = np.squeeze(boxes)
                scores = np.squeeze(scores)
                classes = np.squeeze(classes).astype(np.int32)

                if len(scores) < 1:
                    return TrafficLight.UNKNOWN;
                
                klass = classes[0]
                
                return TLClassifier.class_to_traffic_light(klass)

    @staticmethod
    def class_to_traffic_light(klass):
        rospy.logwarn("klass :%s", klass)
        if klass == 1 or klass == 2:
            return TrafficLight.RED
        elif klass == 3:
            return TrafficLight.GREEN
        else:
            return TrafficLight.UNKNOWN

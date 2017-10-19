from styx_msgs.msg import TrafficLight
import tensorflow as tf
import cv2
import numpy as np
import rospy

class TLClassifier(object):
    def __init__(self):
        self.model = TLClassifier.load_graph()
        self.model.as_default()
        self.sess = tf.Session(graph=self.model)
        self.image_tensor = self.model.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.model.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.model.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.model.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.model.get_tensor_by_name('num_detections:0')


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
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)

        # Actual detection.
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        if len(scores) < 1:
            return TrafficLight.UNKNOWN;

        klass = classes[0]

        return TLClassifier.class_to_traffic_light(klass)


    @staticmethod
    def class_to_traffic_light(klass):
        rospy.logdebug("klass :%s", klass)
        if klass == 1 or klass == 2:
            return TrafficLight.RED
        elif klass == 3:
            return TrafficLight.GREEN
        else:
            return TrafficLight.UNKNOWN

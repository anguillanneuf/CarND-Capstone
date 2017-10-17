#!/usr/bin/env python
import rospy
from dummy_detector import DummyDetector
from real_detector import RealDetector

DUMMY = True

if __name__ == '__main__':
    try:
        if DUMMY:
            detector = DummyDetector()
        else:
            detector = RealDetector()
        detector.loop()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')

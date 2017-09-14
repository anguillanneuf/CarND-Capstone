import math

class MathHelper:
    @staticmethod
    def distance(waypoints, wp1, wp2):
        dist = 0
        for i in range(wp1, wp2+1):
            dist += WaypointUpdater.distance(waypoints[wp1].pose.pose.position,
                                             waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    @staticmethod
    def distance(a, b):
        return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)

import math

class Math3DHelper:
    @staticmethod
    def distance(a, b):
        return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)

import csv
from typing import List
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.colors import ListedColormap

from Point import Point


class Data:

    @staticmethod
    def find_closest( points:List[Point], point: Point, count: int):
        distances = map(
            lambda point: {"point": point, "distance": point.distance(point)}, points)
        distances = sorted(
            distances, key=lambda distance: distance["distance"])
        return list(map(lambda distance: distance["point"], distances[:count]))

    @staticmethod
    def read_points_from_file(file_name: str):
        points: List[Point] = []
        with open(file_name) as dataset_file:
            reader = csv.DictReader(
                dataset_file, fieldnames=('X', 'Y', 'Class'))
            for point in reader:
                points.append(Point(point["X"], point["Y"], point["Class"]))
        return points

    @staticmethod
    def showDataPoints(points: List[Point]):
        print(np.array([str(i) for i in points]).T)

    ##Draws dots according to class
    @staticmethod
    def show_data(points: List[Point]):
        classColormap = ListedColormap(['#FF0000', '#00FF00'])
        pl.scatter([points[i].x for i in range(len(points))],
                [points[i].y for i in range(len(points))],
                c=[points[i].label for i in range(len(points))],
                cmap=classColormap)
        pl.show()

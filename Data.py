from typing import List
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.colors import ListedColormap

from Point import Point
from CsvReader import Csv


class Data:

    @staticmethod
    def read_points_from_file(file_name: str):
        points: List[Point] = []
        reader = Csv.read_data(file_name, fields_names=(
            'X', 'Y', 'Class'), separator="\t", delimiter=",")
        for point in reader:
            points.append(Point(point["X"], point["Y"], point["Class"]))
        return points

    @staticmethod
    def show_data(points: List[Point]):
        print(np.array([str(i) for i in points]).T)

    # Draws dots according to class
    @staticmethod
    def show_colored_data(points: List[Point]):
        classColormap = ListedColormap(['#FF0000', '#00FF00'])
        pl.scatter([points[i].x for i in range(len(points))],
                   [points[i].y for i in range(len(points))],
                   c=[points[i].label for i in range(len(points))],
                   cmap=classColormap)
        pl.show()

    @staticmethod
    def show_difference(test_points: List[Point], predicted_labels: List[Point]):
        classColormap = ListedColormap(['#FF0000', '#00FF00','#F08080','#90EE90'])
        colors=[]

        for i in range(len(predicted_labels)):
            if (predicted_labels[i]==0 and test_points[i].label==0):
                colors.append([0,50])
            if (predicted_labels[i]==1 and test_points[i].label==1):
                colors.append([1,50])
            if (predicted_labels[i]==1 and test_points[i].label==0):
                colors.append([2,50])
            #when it's been predicted as red but it is actually green
            if (predicted_labels[i]==0 and test_points[i].label==1):
                colors.append([3,50])

        pl.scatter([test_points[i].x for i in range(len(test_points))],
                [test_points[i].y for i in range(len(test_points))],
                c=[colors[i][0] for i in range(len(colors))],
                s=[colors[i][1] for i in range(len(colors))],
                cmap=classColormap)
        pl.show()
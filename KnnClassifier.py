from enum import Enum
from typing import List, Dict, Tuple
from Point import Point
from math import exp, pi, sqrt, log


class KernelType(Enum):
    # default linear
    Def = 0
    # оптимальное Епанчиклва
    E = 1
    # квартическое
    Q = 2
    # теругольное
    T = 3
    # гаусовское
    G = 4
    # прямоугольное rectangle
    R = 5

    @staticmethod
    def as_list():
        return [KernelType.Def, KernelType.E, KernelType.Q,  KernelType.T, KernelType.G, KernelType.R]

class CoordinateSystem(Enum):
    # Дикартово пространства
    Cartesian = 0
    # прямоугольное rectangle
    Polar = 1

    @staticmethod
    def as_list():
        return [CoordinateSystem.Cartesian, CoordinateSystem.Polar]


class KnnClassifier:

    train_data: List[Point]

    def train(self, train_data: List[Point], labels, k_neighbors, power=2, kernel_type: KernelType = KernelType.Def, coordinate_system: CoordinateSystem = CoordinateSystem.Cartesian):
        self.train_data = train_data
        self.labels = labels
        self.k_neighbors = k_neighbors
        self.power = power
        self.kernel_type = kernel_type
        self.coordinate_system = coordinate_system

    def predict(self, test_data: List[Point]):
        predictions = []
        for test_point in test_data:
            # Calculate all distances between test point and other points
            neighbors: List[Tuple[Point, float]] = self.__find_closest(test_point)

            measured_labels: List[Tuple[str, float]] = list()
            predicted_label = 0.0
            if self.coordinate_system == CoordinateSystem.Polar:
                for label in self.labels:
                    current_label = [neighbor for neighbor in neighbors if neighbor[0].label == label]
                    weights = self.__get_measured_label_by_proximity(current_label, test_point)
                    measured_labels.append((label, weights))
                predicted_label = max(
                measured_labels, key=lambda measured_label: measured_label[1])[0]
            else:
                for label in self.labels:
                    current_label = [neighbor for neighbor in neighbors if neighbor[0].label == label]
                    weights = self.__get_measured_label_by_weights(current_label)
                    measured_labels.append((label, weights))

                predicted_label = min(
                    measured_labels, key=lambda measured_label: measured_label[1])[0]
            predictions.append(predicted_label)
        return predictions

    def __get_measured_label_by_proximity(self, neighbors: List[Tuple[Point, float]], test_point: Point):
        result = 0.0
        for item in neighbors:
            result += self.__calculate_proximity(item[0], test_point)
        return result

    def __get_measured_label_by_weights(self, neighbors: List[Tuple[Point, float]]):
        result = 0.0
        for item in neighbors:
            #result += item[1] * self.__calculate_kernel(item[1], self.kernel_type)
            result += log(self.__calculate_kernel(item[1], self.kernel_type))
        return result

    # Minkowski
    def __find_closest(self, test_point: Point, power=2):
        distances = map(lambda point: {"point": point, "distance": self.__calculate_distance(
            point, test_point, power)}, self.train_data)
        distances = sorted(
            distances, key=lambda distance: distance["distance"])
        result: List[Tuple[Point, float]] = list(
            map(lambda distance: (distance["point"], distance["distance"]), distances[:self.k_neighbors]))
        return result

    @staticmethod
    def __calculate_distance(point1, point2, p=2):
        return pow(abs(point1.x-point2.x)**p + abs(point1.y-point2.y)**p, 1/p)

    # косинусная мера calculate_cosine_score
    @staticmethod
    def __calculate_proximity(point1: Point, point2: Point):
        def dot_points(point1: Point, point2: Point):
            d = 0.0
            d += point1.x * point2.x + point1.y * point2.y
            return d
        return dot_points(point1, point2) / sqrt(dot_points(point1, point1)) / sqrt(dot_points(point2, point2))

    @staticmethod
    def __calculate_kernel(distance: float, kernel_type: KernelType):
        r = distance
        # оптимальное Епанчиклва
        if kernel_type == KernelType.E:
            return 3/4*(1-r**2)
        # квартическое
        if kernel_type == KernelType.Q:
            return 15/16*(1-r**2)**2
        # теругольное
        if kernel_type == KernelType.T:
            return 1-abs(r)
        # гуасовское
        if kernel_type == KernelType.G:
            return pow(2 * pi, (-0.5))*exp(-0.5*r**2)
        # прямоугольное rectangle
        if kernel_type == KernelType.R:
            return 1/2
        return 1

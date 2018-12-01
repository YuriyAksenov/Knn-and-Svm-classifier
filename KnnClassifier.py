from enum import Enum
from typing import List, Dict, Tuple
from Point import Point
from math import exp, pi


class KernelType(Enum):
    # default
    Def = 0
    # оптимальное Епанчиклва
    E = 0
    # квартическое
    Q = 1
    # теругольное
    T = 2
    # гаусовское
    G = 3
    # прямоугольное rectangle
    R = 4


class KnnClassifier:

    train_data: List[Point]

    def train(self, train_data: List[Point], labels, k_neighbors, power=2, kernel_type: KernelType = KernelType.Def):
        self.train_data = train_data
        self.labels = labels
        self.k_neighbors = k_neighbors
        self.power = power
        self.kernel_type = kernel_type

    def predict(self, test_data: List[Point]):
        predictions = []
        for test_point in test_data:
            # Calculate all distances between test point and other points
            neighbors = self.__find_closest(test_point)

            labels_wights: List[Tuple[str, float]] = list()
            for label in self.labels:
                weights = self.__get_label_weights(
                    [neighbor for neighbor in neighbors if neighbor[1] == label])
                labels_wights.append((label, weights))

            predicted_label = max(
                labels_wights, key=lambda distance: labels_wights[1])[0]
            predictions.append(predicted_label)
        return predictions


    def __get_label_weights(self, neighbors: List[Tuple[Point, float]]):
        result = 0
        for item in neighbors:
            result += self.__calculate_kernel(item[1], self.kernel_type)
        return result

    # Minkowski
    def __find_closest(self, test_point: Point, power=2):
        distances = map(lambda point: {"point": point, "distance": self.__calculate_distance(
            point, test_point, power)}, self.train_data)
        distances = sorted(
            distances, key=lambda distance: distance["distance"])
        result: List[Tuple[Point, float]] = list(
            map(lambda distance: distance["point"], distances[:self.k_neighbors]))
        return result

    @staticmethod
    def __calculate_distance(point1, point2, p=2):
        return pow(abs(point1.x-point2.x)**p + abs(point1.y-point2.y)**p, 1/p)

    @staticmethod
    def __calculate_distance(point1, point2, p=2):
        return pow(abs(point1.x-point2.x)**p + abs(point1.y-point2.y)**p, 1/p)

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

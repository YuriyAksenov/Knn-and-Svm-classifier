from typing import List, Dict

from Data import Data
from Point import Point
from  Logging import Logging
from KnnClassifier import KnnClassifier, KernelType, CoordinateSystem
from Stat import Stat
from Metrics import Metrics
import numpy as np

class DataAnalyzer:

    def __init__(self, points:List[Point], folds_count:int):
        self.folds_count = folds_count
        self.train_data = list()
        self.test_data = list()
        self.make_cross_validation(points, folds_count)

    def make_cross_validation(self, points:List[Point], folds_count: int):
        fold_length = round(len(data) / folds_count)
        for i in range(folds_count):
            n = (folds_count-i) * fold_length
            train_data = data[: n - fold_length] + data[n:]
            test_data = data[n - fold_length : n]
            self.train_data.append(train_data)
            self.test_data.append(test_data)

    # def transformData(point, k1, k2, coorinateSystem):
    #         if "Polar" in coorinateSystem or "polar" in coorinateSystem:
    #         r, angle = translateFromCartesianToPolarCoordinates(point.x, point.y)
    #         r *= k1
    #         angle *= k2
    #         x, y = translateFromPolarToCartesianCoordinates(r, angle)
    #         point.x = x
    #         point.y = y
    #         return

    #     point.x *= k1
    #     point.y *= k2
        
    # def makeSpatialTransformations(data, kx, ky):
    #     for item in data:
    #         item.x *= ky
    #         item.y *= ky
            
    # def makeSpatialTransformationsPolar(data, kr, kangle):
    #     for item in data:
    #         item.r *= kr
    #         item.angle *= kangle
            
    # def translateFromCartesianToPolarCoordinates(x,y):
    #     r = math.sqrt(x**2+y**2)
    #     angle = math.atan2(y, x)
    #     return (r, angle)
        
    # def translateFromPolarToCartesianCoordinates(r, angle):
    #     x = r*math.cos(angle)
    #     y = r*math.sin(angle)
    #     return (x, y)



def calculateAverage(paramsArray, index):
    s = 0
    for item in paramsArray:
        s += item[index]
    return s / len(paramsArray)


fileName = 'chips.txt'
data = readData(fileName)

showData(data)

def analyze_knn(self):

    N= np.sqrt(len(self.test_data[0] + len(self.train_data[0]))) + 1

    numberOfLabels = 2
    numberOfNeighbors = int(N)
    numberOfFolds = int(N*2)
    mumberOfPowers = 3
    kernels = KernelType.as_list()
    coordinateSystems = CoordinateSystem.as_list()
    stats =  [] 

    k1s = [i / 10.0 for i in range(1, 10, 2)] + [float(i) for i in range(1,20,4)];
    k2s = [i / 10.0 for i in range(1, 10, 2)] + [float(i) for i in range(1,20,4)]

    for numFolds in range(2, numberOfFolds):
        for numNeighbor in range(2, numberOfNeighbors):
            for power in range(2, mumberOfPowers): 
                for kernel in kernels:
                    for k1 in k1s:
                        for k2 in k2s:
                            for coordinateSystem in coordinateSystems:                       
                                dataCopy = copy.deepcopy(data)
                                for item in dataCopy:
                                    transformData(item, k1, k2, coordinateSystem)
                                trainDatas, testDatas = getCrossValidation(dataCopy, numFolds)
                                crossParams = []
                                for i in range(len(trainDatas)):
                                    predictions = predict(trainDatas[i], testDatas[i], numberOfLabels, numNeighbor, power, kernel)
                                    params = calculateParameters(testDatas[i], predictions)
                                    crossParams.append(params)
                                avAccuracy = calculateAverage(crossParams, 3)
                                avFscore = calculateAverage(crossParams, 4)
                                stat = Stat(numFolds, numNeighbor, power, kernel, k1, k2, coordinateSystem, avAccuracy, avFscore)
                                print(stat)
                                stats.append(stat)
                        

    print(np.array([str(i) for i in stats]).T)

    print("")
    print("Max accuracy")
    print(max(stats, key=lambda x: x.accuracy))
    print("Max fscore")
    print(max(stats, key=lambda x: x.fscore))
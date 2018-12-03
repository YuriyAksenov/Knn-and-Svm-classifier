from typing import List, Dict

from Data import Data
from Point import Point
from  Logging import Logging
from KnnClassifier import KnnClassifier, KernelType, CoordinateSystem
from Stat import Stat
from Metrics import Metrics
import numpy as np
import copy
import random

class DataAnalyzer:

    @staticmethod
    def make_cross_validation(points:List[Point], folds_count: int):
        fold_length = round(len(points) / folds_count)
        train_data = list()
        test_data = list()
        for i in range(folds_count):
            n = (folds_count-i) * fold_length
            train_fold = points[: n - fold_length] + points[n:]
            test_fold = points[n - fold_length : n]
            train_data.append(train_fold)
            test_data.append(test_fold)
        return (train_data, test_data)

    @staticmethod
    def analyze_knn(data:List[Point]):

        N= np.sqrt(len(data)) + 1
        random.shuffle(data)
        random.shuffle(data)
        random.shuffle(data)
        random.shuffle(data)
        random.shuffle(data)
        random.shuffle(data)


        numberOfLabels = 2
        numberOfNeighbors = int(N)
        numberOfFolds = int(N*2)
        mumberOfPowers = 3
        kernels = KernelType.as_list()
        coordinateSystems = CoordinateSystem.as_list()
        stats =  [] 

        k1s = [i / 10.0 for i in range(1, 10, 2)] + [float(i) for i in range(1,20,4)];
        k2s = [i / 10.0 for i in range(1, 10, 2)] + [float(i) for i in range(1,20,4)]

        dataCopy = copy.deepcopy(data)
        #for item in dataCopy:
        #   transformData(item, k1, k2, coordinateSystem)
        trainData, testData = DataAnalyzer.make_cross_validation(dataCopy, numberOfFolds)

        for numFolds in range(4, numberOfFolds):
            for numNeighbor in range(3, numberOfNeighbors):
                for power in range(2, mumberOfPowers):
                    for coordinateSystem in coordinateSystems:
                        for kernel in kernels:

                            f_scores = []
                            for i in range(len(trainData)):
                                classifier = KnnClassifier()
                                classifier.train(trainData[i], [0,1],numNeighbor,power,kernel,coordinateSystem)
                                test_item = testData[i]
                                predictions = classifier.predict(test_item)
                                f_score = Metrics.f_score([item.label for item in test_item], predictions)
                                f_scores.append(f_score)
                            avFscore = DataAnalyzer.calculateAverage(f_scores)
                            stat = Stat(numFolds, numNeighbor, power, kernel, 0, 0, coordinateSystem, 0, avFscore)
                            print(stat)
                            stats.append(stat)
                            

        print(np.array([str(i) for i in stats]).T)

        print("")
        print("Max f_score general")
        print(max(stats, key=lambda x: x.f_score))
        print("Max f_score Cartesian")
        print(max([stat for stat in stats if stat.coordinateSystem == CoordinateSystem.Cartesian], key=lambda x: x.f_score))
        print("Max f_score Polar")
        print(max([stat for stat in stats if stat.coordinateSystem == CoordinateSystem.Polar], key=lambda x: x.f_score))

    @staticmethod
    def calculateAverage(paramsArray):
        s = 0
        for item in paramsArray:
            s += item
        return s / len(paramsArray)

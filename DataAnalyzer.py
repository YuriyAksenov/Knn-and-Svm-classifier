from typing import List, Dict

from Data import Data
from Point import Point
from  Logging import Logging
from KnnClassifier import KnnClassifier, KernelType, CoordinateSystem
from Stat import Stat, StatSvm
from Metrics import Metrics
import numpy as np
import copy
import random
from SvmClassifier import * 

class DataAnalyzer:

    @staticmethod
    def analyze_svm(data:List[Point]):
        x = np.array([[item.x, item.y] for item in data])
        y = np.array([-1 if item.label == 0 else 1 for item in data])

        x_train = x[:100]
        y_train = y[:100]
        x_test = x[100:]
        y_test = y[100:]

        svm = SvmClassifier(kernel=polynomial_kernel)
        svm.fit(x_train,y_train)
        res = svm.predict(x_test)
        
        stats =  [] 
        kernels = [polynomial_kernel, linear_kernel, gaussian_kernel]
        numberOfFolds = 10
      
        dataCopy = copy.deepcopy(data)
        trainData, testData = DataAnalyzer.make_cross_validation(dataCopy, numberOfFolds)

        for numFolds in range(4, numberOfFolds):
            for kernel in kernels:

                f_scores = []
                p_values = []
                t_test = ""
                for i in range(len(trainData)):

                    x_train = np.array([[item.x, item.y] for item in trainData[i]])
                    y_train = np.array([-1 if item.label == 0 else 1 for item in trainData[i]])

                    x_test = np.array([[item.x, item.y] for item in testData[i]])
                    y_test = np.array([-1 if item.label == 0 else 1 for item in testData[i]])

                    classifier = SvmClassifier(kernel=kernel)
                    classifier.fit(x_train,y_train)
                    predictions = svm.predict(x_test)


                    f_score = Metrics.f_score([item.label for item in testData[i]], predictions)
                    f_scores.append(f_score)
                    p_value = Metrics.p_value([item.label for item in testData[i]], predictions, 2)
                    p_values.append(p_value[1])
                    t_test = Metrics.t_test([item.label for item in testData[i]], predictions)
                avFscore = DataAnalyzer.calculateAverage(f_scores)
                avPvalue = DataAnalyzer.calculateAverage(p_values)
                stat = StatSvm(numFolds, kernel, avFscore, avPvalue, t_test)
                #print(stat)
                stats.append(stat)



        print("SVM ---------------------------------------------------------")
        print("")
        print("Max f_score general")
        print(max(stats, key=lambda x: x.f_score))
        print("Max p value general")
        print(max(stats, key=lambda x: x.p_value))
        print("Max t Wilcoxon general")
        print(max(stats, key=lambda x: x.p_value))
        print("")
        print("SVM ---------------------------------------------------------")


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
        


        numberOfLabels = 2
        numberOfNeighbors = 10
        numberOfFolds = 10
        mumberOfPowers = 3
        kernels = KernelType.as_list()
        coordinateSystems = CoordinateSystem.as_list()
        stats =  [] 

        power = 2

        dataCopy = copy.deepcopy(data)
        trainData, testData = DataAnalyzer.make_cross_validation(dataCopy, numberOfFolds)

        for numFolds in range(4, numberOfFolds):
            for numNeighbor in range(3, numberOfNeighbors):
                #for power in range(2, mumberOfPowers):
                for coordinateSystem in coordinateSystems:
                    for kernel in kernels:

                        f_scores = []
                        p_values = []
                        t_test = ""
                        for i in range(len(trainData)):
                            classifier = KnnClassifier()
                            classifier.train(trainData[i], [0,1],numNeighbor,power,kernel,coordinateSystem)
                            test_item = testData[i]
                            predictions = classifier.predict(test_item)
                            f_score = Metrics.f_score([item.label for item in test_item], predictions)
                            f_scores.append(f_score)
                            p_value = Metrics.p_value([item.label for item in test_item], predictions, 2)
                            p_values.append(p_value[1])
                            t_test = Metrics.t_test([item.label for item in test_item], predictions)
                        avFscore = DataAnalyzer.calculateAverage(f_scores)
                        avPvalue = DataAnalyzer.calculateAverage(p_values)
                        stat = Stat(numFolds, numNeighbor, power, kernel, coordinateSystem, avFscore, avPvalue, t_test)
                        #print(stat)
                        stats.append(stat)
                            

        #print(np.array([str(i) for i in stats]).T)

        print("KNN ---------------------------------------------------------")
        print("")
        print("Max f_score general")
        print(max(stats, key=lambda x: x.f_score))
        print("Max p value general")
        print(max(stats, key=lambda x: x.p_value))
        print("Max t Wilcoxon general")
        print(max(stats, key=lambda x: x.p_value))
        print("")
        print("KNN ---------------------------------------------------------")

    @staticmethod
    def calculateAverage(paramsArray):
        s = 0
        for item in paramsArray:
            s += item
        return s / len(paramsArray)

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
        print("Max accuracy")
        print(max(stats, key=lambda x: x.accuracy))
        print("Max fscore")
        print(max(stats, key=lambda x: x.f_score))

    @staticmethod
    def calculateAverage(paramsArray):
        s = 0
        for item in paramsArray:
            s += item
        return s / len(paramsArray)

#     def run_knn():
#         data = Data()
#     data.read_data("./Dataset")

#     folds = list(data.parts.keys())
#     f_scores = []

#     global_test_data = []
#     global_predicted_data = []

#     for test_fold in folds:
#         test_data = []
#         predicted_data = []

#         print("test_fold: " + test_fold +
#               "-------------------------------------------------------------------")
#         analizator = MailComplexAnalizator()
#         for train_fold in folds:
#             if train_fold != test_fold:
#                 analizator.add_to_word_statistics(data.parts[train_fold])
#         test_mails: DataPart = data.parts[test_fold]

#         print("test_fold: " + test_fold +
#               " calculate fscore ---------------------------------------------------")

#         for spam_mail in test_mails.spams:
#             test_data.append(1)
#             global_test_data.append(1)
#             is_spam = int(analizator.is_spam(spam_mail, is_check_incomings=True,
#                                              accounting_ratio_subject=1, accounting_ratio_body=1, accounting_ratio_words=1))
#             predicted_data.append(is_spam)
#             global_predicted_data.append(is_spam)

#         for ham_mail in test_mails.hams:
#             test_data.append(0)
#             global_test_data.append(0)
#             is_spam = int(analizator.is_spam(ham_mail, is_check_incomings=True,
#                                              accounting_ratio_subject=1, accounting_ratio_body=1, accounting_ratio_words=1))
#             predicted_data.append(is_spam)
#             global_predicted_data.append(is_spam)

#         fscore = Metrics.f_score(test_data, predicted_data)
#         print("test_fold: " + test_fold + " calculated fscore: " +
#               str(fscore) + "---------------------------------------------------")
#         f_scores.append(fscore)

#     #print(" ".join(map(str, f_scores)))
#     f_score_average = float(sum(f_scores)) / float(len(f_scores))

#     print("fscore:" + str(f_score_average))

#     print("------- confusion matrix ------")
#     Metrics.plot_confusion_matrix(global_test_data, global_predicted_data)

#     count = 0
#     print(len(global_predicted_data))
#     print(len(global_test_data))

#     for i in range(len(global_test_data)):
#         if(global_predicted_data[i] != global_test_data[i]):
#             count += 1
#     print("Eror percent: " + str(count/len(global_predicted_data)))
#     print(count)
#     print(len(global_predicted_data))



# # можно использовать ещё окно контекста для понимания отношения слов word2vec

# print("Hello world")

from DataAnalyzer import DataAnalyzer
from Data import Data
from Svm import *
import  numpy as np
import random
from Metrics import * 

fileName = 'Chips.txt'
data = Data.read_points_from_file(fileName)
# Data.show_data(data)
DataAnalyzer.analyze_knn(data)
# random.shuffle(data)
# random.shuffle(data)
# random.shuffle(data)
# random.shuffle(data)
# random.shuffle(data)


# x = np.array([[item.x, item.y] for item in data])
# y = np.array([-1 if item.label == 0 else 1 for item in data])

# x_train = x[:100]
# y_train = y[:100]
# x_test = x[100:]
# y_test = y[100:]

# svm = SvmClassifier(C=1, kernel=polynomial_kernel)
# svm.fit(x_train,y_train)
# res = svm.predict(x_test)

# res = np.array([0 if item == -1 else 1 for item in res])
# y_test = np.array([0 if item == -1 else 1 for item in y_test])

# print(res)
# f = Metrics.f_score(y_test, res);

# print('-------------------------------')
# print(f)

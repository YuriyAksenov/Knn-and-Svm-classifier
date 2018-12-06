
from DataAnalyzer import DataAnalyzer
from Data import Data
from SvmClassifier import *
import  numpy as np
import random
from Metrics import * 
from KnnClassifier import * 
from SvmClassifier import * 



# можно увеличить простарнства в knn путем применения лангранжа. А потом при предикте брать те веса, которые посчитали для лангранжа, точку проецировать. И брать соседей




fileName = 'Chips.txt'
data = Data.read_points_from_file(fileName)
random.shuffle(data)
random.shuffle(data)
random.shuffle(data)
random.shuffle(data)
random.shuffle(data)
random.shuffle(data)
# Data.show_data(data)
fknn, knnPredict = DataAnalyzer.analyze_knn_one(data, numFolds = 10)

fsvm, svmPredict = DataAnalyzer.analyze_svm_one(data, numFolds =10)
print("fscores ---------------   ")
print(fknn)
print(fsvm)

test = Metrics.t_test_empirical(fknn, fsvm)
print("TEST ---------------   ")
print(test)

print("p value ---------------   ")
p = Metrics.p_value(knnPredict, svmPredict, 2)
print(p)







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











svm_print = """
SVM ---------------------------------------------------------

Max f_score general
| numFold:4 | kernel:<function gaussian_kernel at 0x00000225CC41D400> | 0.6008 | 0.7325 | improved|
Max p value general
| numFold:4 | kernel:<function gaussian_kernel at 0x00000225CC41D400> | 0.6008 | 0.7325 | improved|
Max t Wilcoxon general
| numFold:4 | kernel:<function gaussian_kernel at 0x00000225CC41D400> | 0.6008 | 0.7325 | improved|

SVM ---------------------------------------------------------
"""

knn_print = """
KNN ---------------------------------------------------------

Max f_score general
| numFold:4 | numNeighbor:6 | power:2 | kernel:KernelType.E | CoordinateSystem.Cartesian | 0.6989 | 0.9000 | improved |
Max p value general
| numFold:4 | numNeighbor:6 | power:2 | kernel:KernelType.E | CoordinateSystem.Cartesian | 0.6989 | 0.9000 | improved |
Max t Wilcoxon general
| numFold:4 | numNeighbor:6 | power:2 | kernel:KernelType.E | CoordinateSystem.Cartesian | 0.6989 | 0.9000 | improved |

KNN ---------------------------------------------------------
"""

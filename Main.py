
from DataAnalyzer import DataAnalyzer
from Data import Data


fileName = 'Chips.txt'
data = Data.read_points_from_file(fileName)
Data.show_data(data)
DataAnalyzer.analyze_knn(data)




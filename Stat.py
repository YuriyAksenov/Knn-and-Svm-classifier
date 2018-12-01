class Stat:
    def __init__(self, numFolds, numNeighbor, power, kernel, k1, k2, coordinateSystem, accuracy, f_score):
        self.numFolds = numFolds
        self.numNeighbor = numNeighbor
        self.kernel = kernel
        self.power = power
        self.k1 = k1
        self.k2 = k2
        self.coordinateSystem = coordinateSystem
        self.accuracy = accuracy
        self.f_score = f_score
    
    # def __str__(self):
    #     main = "| %d | %d | %d | %s | %0.4f | %0.4f | %s " % (self.numFolds, self.numNeighbor, self.power, self.kernel, self.k1, self.k2, self.coordinateSystem)
    #     params = "| %0.4f | %0.4f |" % (self.accuracy, self.f_score)
    #     return  main + params 
    def __str__(self):
        main = "| numFold:%d | numNeighbor:%d | power:%d | kernel:%s | %s " % (self.numFolds, self.numNeighbor, self.power, self.kernel, self.coordinateSystem)
        params = "| %0.4f |" % (self.f_score)
        return  main + params 
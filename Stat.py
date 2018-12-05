class Stat:
    def __init__(self, numFolds, numNeighbor, power, kernel, coordinateSystem, f_score, p_value, t_test):
        self.numFolds = numFolds
        self.numNeighbor = numNeighbor
        self.kernel = kernel
        self.power = power
        self.coordinateSystem = coordinateSystem
        self.f_score = f_score
        self.p_value = p_value
        self.t_test = t_test
    
    # def __str__(self):
    #     main = "| %d | %d | %d | %s | %0.4f | %0.4f | %s " % (self.numFolds, self.numNeighbor, self.power, self.kernel, self.k1, self.k2, self.coordinateSystem)
    #     params = "| %0.4f | %0.4f |" % (self.accuracy, self.f_score)
    #     return  main + params 
    def __str__(self):
        main = "| numFold:%d | numNeighbor:%d | power:%d | kernel:%s | %s " % (self.numFolds, self.numNeighbor, self.power, self.kernel, self.coordinateSystem)
        params = "| %0.4f | %0.4f | %0.4f |" % (self.f_score, self.p_value, self.t_test)
        return  main + params 

class StatSvm:
    def __init__(self, numFolds, kernel, f_score, p_value, t_test):
        self.numFolds = numFolds
        self.kernel = kernel
        self.f_score = f_score
        self.p_value = p_value
        self.t_test = t_test
    
    # def __str__(self):
    #     main = "| %d | %d | %d | %s | %0.4f | %0.4f | %s " % (self.numFolds, self.numNeighbor, self.power, self.kernel, self.k1, self.k2, self.coordinateSystem)
    #     params = "| %0.4f | %0.4f |" % (self.accuracy, self.f_score)
    #     return  main + params 
    def __str__(self):
        main = "| numFold:%d | kernel:%s " % (self.numFolds, self.kernel)
        params = "| %0.4f | %0.4f | %0.4f |" % (self.f_score, self.p_value, self.t_test)
        return  main + params 
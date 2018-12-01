class Stat:
    def __init__(self, folders_count: int, neighbors_count, power, kernel, k1, k2, coordinate_system, accuracy, f_score):
        self.folders_count = folders_count
        self.neighbors_count = neighbors_count
        self.kernel = kernel
        self.power = power
        self.k1 = k1
        self.k2 = k2
        self.coordinate_system = coordinate_system
        self.accuracy = accuracy
        self.f_score = f_score

    def __str__(self):
        main = "| %d | %d | %d | %s | %0.4f | %0.4f | %s " % (
            self.folders_count, self.neighbors_count, self.power, self.kernel, self.k1, self.k2, self.coordinate_system)
        params = "| %0.4f | %0.4f |" % (self.accuracy, self.f_score)
        return main + params

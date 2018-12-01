class Point:
    
    x: float
    y: float
    category: int

    def __init__(self, x: float, y: float, category: int):
        self.x = float(x)
        self.y = float(y)
        self.category = category

    def get_distance(self, other_point: Point):
        return ((self.x - other_point.x) ** 2 + (self.y - other_point.y) ** 2) ** 0.5

    def __str__(self):
        return "{x:%f,y:%f,c:%s}" % (self.x, self.y, self.category)
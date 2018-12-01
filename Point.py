class Point:

    x: float
    y: float
    label: int

    def __init__(self, x: float, y: float, label: int):
        self.x = float(x)
        self.y = float(y)
        self.label = label

    def __str__(self):
        return "{x:%f,y:%f,c:%s}" % (self.x, self.y, self.label)

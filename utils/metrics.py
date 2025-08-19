class metrics(object):
    def __init__(self):
        self.value = 0
        self.size = 0
        self.average = 0
        self.sum = 0
        self.hist = []

    def reset(self):
        self.value = 0
        self.size = 0
        self.average = 0
        self.sum = 0

    def step(self, value, count):
        self.value = value
        self.size += count
        self.sum += value * count
        self.average = self.sum / self.size

    def history(self):
        self.hist.append(self.average)
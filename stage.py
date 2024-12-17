import math
from constants import c

class Stage1D:
    def __init__(self, width=5, w_h=10000):
        self.width = width
        self.w_h = w_h
        self.grid = [complex(0)] * self.w_h
        # self.previous_grid = [complex(0)] * self.w_h
        self.samples = []

    def add_sample(self, sample: complex, position: float, direction: int = 1):
        self.samples.append((sample, position, direction))

    def next_step(self):
        delta_t = self.width/self.w_h/c
        v = c
        # Clear the grid
        self.grid = [complex(0)] * self.w_h
        # Update locations
        for i in range(len(self.samples)):
            sample, position, direction = self.samples[i]
            index = int(math.floor(position / self.width * self.w_h)) + direction
            if 0 < index < len(self.grid):
                self.grid[index] = sample
            self.samples[i] = (sample, position+c*delta_t, direction)
        return self.grid




# class Stage2D:
#     def __init__(self,  width=5): # Add constraints
#         self.grid = [[]*width]
#     def add_sample(self, sample: IQSample):

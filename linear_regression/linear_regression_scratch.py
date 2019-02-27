from math import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random
style.use('fivethirtyeight')


class Functions:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys
        self.length = len(self.xs)
        # initializes variables for averages
        self.sum_x = 0
        self.sum_y = 0
        self.mean_x = 0
        self.mean_y = 0
        # initializes variables for r and least square's line (line of best fit)
        self.num_data = 0
        self.x_square = 0
        self.y_square = 0
        self.xy = 0
        self.r = 0
        self.m = 0
        self.b = 0

    def lin_reg(self):
        for i in range(self.length):
            self.xy += int(self.xs[i]) * int(self.ys[i])
            self.sum_x += int(self.xs[i])
            self.sum_y += int(self.ys[i])
            self.x_square += int(self.xs[i]) ** 2
            self.y_square += int(self.ys[i]) ** 2

    def slope(self):
        self.lin_reg()
        self.m = ((self.length * self.xy - self.sum_x * self.sum_y) / (self.length * self.x_square - self.sum_x ** 2))
        return self.m

    def y_int(self):
        self.slope()
        self.b = ((self.sum_y - self.m * self.sum_x) / self.length)
        return self.b

    def pearson(self):
        self.lin_reg()
        self.r = ((self.length * self.xy) - (self.sum_x * self.sum_y)) / sqrt(((self.length * self.x_square) - self.sum_x ** 2) * ((self.length * self.y_square) - self.sum_y ** 2))
        return self.r


def create_data_set(num, variance, step=2, correlation='pos'):
    val = 1
    ys = []
    for i in range(num):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


if __name__ == '__main__':
    # xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
    # ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)

    xs, ys = create_data_set(40, 10, 2, correlation='pos')

    m = Functions(xs, ys).slope()
    b = Functions(xs, ys).y_int()
    r = Functions(xs, ys).pearson()
    r2 = r**2

    print(m, b, r, r2)
    reg_line = [(m*x) + b for x in xs]
    predict_x = 8
    predict_y = (m*predict_x + b)

    plt.scatter(xs, ys)
    plt.scatter(predict_x, predict_y, color='g')
    plt.plot(xs, reg_line)
    plt.show()

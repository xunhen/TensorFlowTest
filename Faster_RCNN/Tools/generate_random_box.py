from numpy import random
import numpy as np


def random_rpn(gts, number=50):
    x_min = []
    y_min = []
    x_max = []
    y_max = []
    for gt in gts:
        for i in range(number):
            x_ = random.uniform(0.8, 1.8)
            y_ = random.uniform(0.8, 1.8)
            y_add = (gt[2] - gt[0]) * y_ / 2
            x_add = (gt[3] - gt[1]) * x_ / 2

            y_offset_random = (gt[2] - gt[0]) * (y_ - 1) / 2
            x_offset_random = (gt[3] - gt[1]) * (x_ - 1) / 2

            offset_y = random.uniform(0, y_offset_random)
            offset_x = random.uniform(0, x_offset_random)

            center_y = (gt[2] - gt[0]) / 2 + gt[0] + offset_y
            center_x = (gt[3] - gt[1]) / 2 + gt[1] + offset_x
            x_min.append(max(center_x - x_add, 0))
            y_min.append(max(center_y - y_add, 0))
            x_max.append(min(center_x + x_add, 1))
            y_max.append(min(center_y + y_add, 1))
    return [y_min, x_min, y_max, x_max]


if __name__ == '__main__':
    rpn = random_rpn(zip([16 / 153], [27 / 334], [91 / 153], [82 / 334]))
    rpn = [[ymin * 469, xmin * 1024, ymax * 469, xmax * 1024] for (ymin, xmin, ymax, xmax) in zip(*rpn)]
    rpn = np.array(rpn)
    rpn = rpn.reshape([1, -1, 4])
    print(rpn)

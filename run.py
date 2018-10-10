#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: dell
# Created Time: 2018-10-10 22:14:00

import numpy as np

from solver11849180 import ParityNGenerator, ForwardArtificialNeuralNectwork


def ann_run():
    weights_out = '''-42.8  -32.4   0      -5.6  -23.6  -33.6  -33.6   41.6      0      0     0
                             -75.3  -32.0   43.2  -41.1  -34.5  -34.8  -34.8   39.8  -58.9      0     0
                             -85.0  -28.1   28.6  -28.0  -28.0  -28.2  -28.2   29.3  -47.6  -41.3     0
                              59.9   12.9  -13.5   13.0   13.0   13.0   13.0  -13.4      0      0  81.8
                          '''
    weights = np.fromstring(weights_out, sep=' ').reshape(4, -1)
    ann = ForwardArtificialNeuralNectwork(7, 7, 1)
    ann.construct(weights)
    genp7 = ParityNGenerator(7)
    vec, ans = genp7(127)
    result = ann.evaluate(vec)
    print(result)


if __name__ == '__main__':
    ann_run()

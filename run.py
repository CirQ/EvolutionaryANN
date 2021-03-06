#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: dell
# Created Time: 2018-10-10 22:14:00

import numpy as np

from solver11849180 import ParityNGenerator, ForwardArtificialNeuralNectwork, EPNet
from util import annotated_timer


@annotated_timer('run bp then sa')
def ann_run():
    ann = ForwardArtificialNeuralNectwork(5, 5, 1)
    ann.initialize(2, 1.0, 0.0, 1.0)
    genp5 = ParityNGenerator(5)
    _, res, vec = map(np.array, zip(*genp5.all()))

    ann.train(vec, res, lr=0.3, epoch=1000)

    result = (ann.evaluate(vec) > 0.5)
    for y, yhat in zip(res, result):
        if y != yhat:
            print(y, yhat)

    print()

    ann.simul_anneal(vec, res, temperature=1.0, steps=1000)

    result = (ann.evaluate(vec) > 0.5)
    for y, yhat in zip(res, result):
        if y != yhat:
            print(y, yhat)


@annotated_timer('test epnet')
def epnet_run():
    epnet = EPNet(10, 5, 5, 1)
    genp5 = ParityNGenerator(5)
    _, res, vec = map(np.array, zip(*genp5.all()))
    top, i = epnet.run(vec, res, num_hid=2, lr=0.5, temperature=1.0)
    print(top)
    print(i)



if __name__ == '__main__':
    epnet_run()
    # ann_run()

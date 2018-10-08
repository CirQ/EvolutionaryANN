#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: dell
# Created Time: 2018-09-25 09:59:21

import numpy as np

from util import *
from solver11849180 import ParityNGenerator, ForwardArtificialNeuralNectwork



import unittest


class ParityNTest(unittest.TestCase):
    def test_constructor(self):
        pn = ParityNGenerator(6)
        self.assertEqual(pn.bound, 64)
        with self.assertRaises(ParityNGenerator.ParityNException):
            ParityNGenerator(-1)

    def test_call(self):
        pn = ParityNGenerator(4)
        self.assertTupleEqual(tuple(pn(11)), (True, False, True, True))
        with self.assertRaises(ParityNGenerator.ParityNException):
            pn(16)
        with self.assertRaises(ParityNGenerator.ParityNException):
            pn(4, 1, a=3)

    def test_all(self):
        it = ParityNGenerator(2).all()
        _, out, vec = next(it)
        self.assertEqual(out, True)
        self.assertTupleEqual(tuple(vec), (False, False))
        _, out, vec = next(it)
        self.assertEqual(out, False)
        self.assertTupleEqual(tuple(vec), (False, True))
        _, out, vec = next(it)
        self.assertEqual(out, False)
        self.assertTupleEqual(tuple(vec), (True, False))
        _, out, vec = next(it)
        self.assertEqual(out, True)
        self.assertTupleEqual(tuple(vec), (True, True))


class FANNTest(unittest.TestCase):
    def test_initialize(self):
        # TODO: test initialize method
        pass

    def test_construct_general(self):
        ann = ForwardArtificialNeuralNectwork(3, 2, 2)
        weights = [[1,3,2,1,0,0],
                   [5,4,3,2,2,0],
                   [2,1,4,4,1,2]]
        expect_weight = [[0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,0,0],
                         [1,3,2,1,0,0,0,0],
                         [0,0,0,0,0,0,0,0],
                         [5,4,3,2,2,0,0,0],
                         [2,1,4,4,1,0,2,0]]
        expect_connectivity = [[False, False, False, False, False, False, False, False],
                               [False, False, False, False, False, False, False, False],
                               [False, False, False, False, False, False, False, False],
                               [False, False, False, False, False, False, False, False],
                               [True,  True,  True,  True,  False, False, False, False],
                               [False, False, False, False, False, False, False, False],
                               [True,  True,  True,  True,  True,  False, False, False],
                               [True,  True,  True,  True,  True,  False, True,  False]]
        expect_hidden = [True, False]
        ann.construct(np.array(weights))
        self.assertListEqual(ann.weight.tolist(), expect_weight)
        self.assertListEqual(ann.connectivity.tolist(), expect_connectivity)
        self.assertListEqual(ann.hidden.tolist(), expect_hidden)

    def test_construct_nparity(self):
        ann = ForwardArtificialNeuralNectwork(3, 3, 1)
        weights = [[1,3,2,1,0],
                   [5,4,3,2,2]]
        expect_weight = [[0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,0,0],
                         [1,3,2,1,0,0,0,0],
                         [0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,0,0],
                         [5,4,3,2,2,0,0,0]]
        expect_connectivity = [[False, False, False, False, False, False, False, False],
                               [False, False, False, False, False, False, False, False],
                               [False, False, False, False, False, False, False, False],
                               [False, False, False, False, False, False, False, False],
                               [True,  True,  True,  True,  False, False, False, False],
                               [False, False, False, False, False, False, False, False],
                               [False, False, False, False, False, False, False, False],
                               [True,  True,  True,  True,  True,  False, False,  False]]
        expect_hidden = [True, False, False]
        ann.construct(np.array(weights))
        self.assertListEqual(ann.weight.tolist(), expect_weight)
        self.assertListEqual(ann.connectivity.tolist(), expect_connectivity)
        self.assertListEqual(ann.hidden.tolist(), expect_hidden)

    def test_evaluate_given7(self):
        weights_out = '''-42.8  -32.4   0      -5.6  -23.6  -33.6  -33.6   41.6      0      0     0
                         -75.3  -32.0   43.2  -41.1  -34.5  -34.8  -34.8   39.8  -58.9      0     0
                         -85.0  -28.1   28.6  -28.0  -28.0  -28.2  -28.2   29.3  -47.6  -41.3     0
                          59.9   12.9  -13.5   13.0   13.0   13.0   13.0  -13.4      0      0  81.8
                      '''
        weights = np.fromstring(weights_out, sep=' ').reshape(4, -1)
        genp7 = ParityNGenerator(7)
        ann = ForwardArtificialNeuralNectwork(7, 7, 1)
        ann.construct(weights)
        for _, res, vec in genp7.all():
            result = ann.evaluate(vec)
            self.assertEqual(result[0]>0.5, not res)    # use not res since the condition is negating in paper

    def test_evaluate_given8(self):
        weights_out = '''-12.4   25.2   27.7  -29.4  -28.9  -29.7  -25.4  -28.5   27.8      0      0     0
                         -40.4   19.6   18.9  -18.1  -19.1  -18.5  -17.3  -18.8   20.4  -67.6      0     0
                         -48.1   16.0   16.1  -15.9  -16.3  -15.8  -15.9  -15.8   16.7  -55.0  -26.7     0
                          45.7  -10.0  -11.0   10.0    9.9    9.4   10.0    9.6  -11.4    6.8    2.3  76.3
                      '''
        weights = np.fromstring(weights_out, sep=' ').reshape(4, -1)
        genp8 = ParityNGenerator(8)
        ann = ForwardArtificialNeuralNectwork(8, 8, 1)
        ann.construct(weights)
        for _, res, vec in genp8.all():
            result = ann.evaluate(vec)
            self.assertEqual(result[0]>0.5, not res)

    def test_tostr(self):
        expected_out = ''' -42.8  -32.4    0.0   -5.6  -23.6  -33.6  -33.6   41.6    0.0    0.0    0.0
 -75.3  -32.0   43.2  -41.1  -34.5  -34.8  -34.8   39.8  -58.9    0.0    0.0
 -85.0  -28.1   28.6  -28.0  -28.0  -28.2  -28.2   29.3  -47.6  -41.3    0.0
  59.9   12.9  -13.5   13.0   13.0   13.0   13.0  -13.4    0.0    0.0   81.8'''
        weights = np.fromstring(expected_out, sep=' ').reshape(4, -1)
        ann = ForwardArtificialNeuralNectwork(7, 7, 1)
        ann.construct(weights)
        self.assertEqual(str(ann), expected_out)

    def test_torepr(self):
        weights_out = '''-12.4   25.2   27.7  -29.4  -28.9  -29.7  -25.4  -28.5   27.8      0      0     0
                         -40.4   19.6   18.9  -18.1  -19.1  -18.5  -17.3  -18.8   20.4  -67.6      0     0
                         -48.1   16.0   16.1  -15.9  -16.3  -15.8  -15.9  -15.8   16.7  -55.0  -26.7     0
                          45.7  -10.0  -11.0   10.0    9.9    9.4   10.0    9.6  -11.4    6.8    2.3  76.3'''
        expected_out = '''   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
 -12.4   25.2   27.7  -29.4  -28.9  -29.7  -25.4  -28.5   27.8    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
 -40.4   19.6   18.9  -18.1  -19.1  -18.5  -17.3  -18.8   20.4  -67.6    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
 -48.1   16.0   16.1  -15.9  -16.3  -15.8  -15.9  -15.8   16.7  -55.0  -26.7    0.0    0.0    0.0    0.0    0.0    0.0    0.0
   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
   0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
  45.7  -10.0  -11.0   10.0    9.9    9.4   10.0    9.6  -11.4    6.8    2.3   76.3    0.0    0.0    0.0    0.0    0.0    0.0'''
        weights = np.fromstring(weights_out, sep=' ').reshape(4, -1)
        ann = ForwardArtificialNeuralNectwork(8, 8, 1)
        ann.construct(weights)
        self.assertEqual(repr(ann), expected_out)


def ann_init():
    weights_out = '''-42.8  -32.4   0      -5.6  -23.6  -33.6  -33.6   41.6      0      0     0
                     -75.3  -32.0   43.2  -41.1  -34.5  -34.8  -34.8   39.8  -58.9      0     0
                     -85.0  -28.1   28.6  -28.0  -28.0  -28.2  -28.2   29.3  -47.6  -41.3     0
                      59.9   12.9  -13.5   13.0   13.0   13.0   13.0  -13.4      0      0  81.8
                  '''
    weights = np.fromstring(weights_out, sep=' ').reshape(4, -1)
    ann = ForwardArtificialNeuralNectwork(7, 7, 1)
    ann.construct(weights)







if __name__ == '__main__':
    unittest.main()
    # ann_init()

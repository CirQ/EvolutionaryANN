#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: cirq
# Created Time: 2018-09-23 18:07:29

import argparse

import numpy as np



class ParityNGenerator(object):
    """ Class for generating Parity-N data set
    """
    class ParityNException(Exception):
        pass

    def __init__(self, N=5):
        """ Constructor

        :param N: parameter in Parity-N problem
        :type N: int
        """
        if not (isinstance(N, int) and N > 0):
            raise self.ParityNException('N can only be positive integer')
        if N > 16:
            raise self.ParityNException("don't use N more than 16")
        self.N = N
        self.bound = 2**N

    def __call__(self, *args, **kwargs):
        """ Get a binary representation of integer num
        """
        if not (len(args)==1 and not bool(kwargs)):
            raise self.ParityNException('only one argument is required')
        num = args[0]
        if not (isinstance(num, int) and 0 <= num < self.bound):
            raise self.ParityNException('num should be an integer within [0,{})'.format(self.bound))
        vec_num = np.zeros(self.N, dtype=np.bool)
        i = self.N - 1
        while num > 0:
            vec_num[i] = num%2
            num, i = num//2, i-1
        return vec_num

    def all(self):
        """ Return all binary vectors within self.bound

        :return: iterable (num, vec) pairs
        :rtype: generator
        """
        for num in range(self.bound):
            vec = self(num)
            out = vec.sum() % 2 == 0
            yield num, out, self(num)



class ForwardArtificialNeuralNectwork(object):
    """ Class of forward ANN.
    """
    class ANNException(Exception):
        pass


    @staticmethod
    def __sigmoid(z, derivative=False):
        """ The sigmoid function and its derivative

        :param z: the parameter
        :type z: number
        :param derivative: whether to calculate derivative
        :type derivative: bool
        :return: the sigmoid value, or its derivative
        :rtype: float
        """
        if derivative:
            return z * (1 - z)
        else:
            return 1 / (1 + np.exp(-z))


    def __init__(self, m_in, n_hid, n_out):
        """ Constructor of a forward ANN instance.

        :param m_in: the dimension of input vector
        :type m_in: int
        :param n_hid: the max-allowed dimension of hidden layer
        :type n_hid: int
        :param n_out: the dimension of output vector
        :type n_out: int
        """
        self.dim_in = m_in + 1  # one extra input dim for bias
        self.dim_hid = n_hid
        self.dim_out = n_out
        self.dim_node = self.dim_in + self.dim_hid + self.dim_out

        self.weight = np.zeros((self.dim_node, self.dim_node), dtype=np.float64)
        self.connectivity = np.zeros((self.dim_node, self.dim_node), dtype=np.bool)
        self.hidden = np.zeros(self.dim_hid, dtype=np.bool)


    def __str__(self):
        # TODO: the string representation of weights
        pass


    def initialize(self):
        """ Initialize the ANN according to the rule specified in the paper.

        :return: None
        :rtype: None
        """
        # TODO: to initiate the matrices
        pass


    def construct(self, weights):
        """ Construct the ANN from output weights specified by the assignment.

        :param weights: a weight matrix in the format specified by the assignment
        :type weights: np.ndarray

        :return: None
        :rtype: None
        """
        din, dout, dhid = self.dim_in, self.dim_out, self.dim_hid   # the max dim
        hid = weights.shape[0] - dout                               # this hidden dim
        if not (weights.shape[1]-din-dout==hid and (0<hid<=dhid)):
            raise self.ANNException('weight matrix hidden nodes not matching')
        if not (dout < weights.shape[0] <= hid+dout):
            raise self.ANNException('weight matrix row shape not matching')
        if not (din+dout < weights.shape[1] <= din+hid+dout):
            raise self.ANNException('weight matrix column shape not matching')

        self.weight[din:din+hid,:din+hid] = weights[:hid,:din+hid]
        self.weight[din:din+hid,din+dhid:] = weights[:hid,din+hid:]
        self.weight[din+dhid:,:din+hid] = weights[hid:,:din+hid]
        self.weight[din+dhid:,din+dhid:] = weights[hid:,din+hid:]

        for i in range(hid):
            self.connectivity[din+i,:din+i] = True
        self.connectivity[din+dhid:,:din+hid] = True
        for i in range(dout):
            self.connectivity[din+dhid+i,din+dhid:din+dhid+i] = True

        self.hidden[:hid] = True


    def evaluate(self, x):
        """ Calculate the output of ANN for input x.

        :param x: the input vector
        :type x: np.ndarry
        :return: the value of evaluated ANN
        :rtype: np.ndarry
        """
        if x.shape != (self.dim_in-1,):
            raise self.ANNException('input dimension not matching')
        bias = np.ones(1)
        tail = np.zeros(self.dim_hid+self.dim_out)
        x = np.concatenate((bias, x, tail), axis=0)

        weight = self.weight * self.connectivity
        for i in range(self.dim_in, self.dim_in+self.dim_hid+self.dim_out):
            net = x.dot(weight[i])
            x[i] = self.__sigmoid(net)
        return x[-self.dim_out:]


    def backpropagate(self, lr):
        # TODO: bp algorithm
        pass








class SimulatedAnnealingSolver(object):
    def __init__(self):
        pass



class EvolutionaryProgramming(object):
    def __init__(self):
        pass



class NParityProblem(object):
    def __init__(self):
        pass



def main():
    parser = argparse.ArgumentParser(description='An N-parity problem solver based on evolutionary ANN')
    parser.add_argument('-s', type=int, required=True, help='An integer random seed', metavar='SEED', dest='seed')
    parser.add_argument('-n', default=5, type=int, help='The parameter N of N-parity problem', metavar='N', dest='n')
    args = parser.parse_args()

    np.random.seed(args.seed)
    n = args.n


if __name__ == '__main__':
    main()

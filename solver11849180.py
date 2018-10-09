#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: cirq
# Created Time: 2018-09-23 18:07:29

import argparse
import io

import numpy as np



class ParityNGenerator(object):
    """ Class for generating Parity-N data set
    """
    class ParityNException(Exception):
        pass

    def __init__(self, N):
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

        :return: iterable (num, ans, vec) pairs
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
        """ Return the weight matrix string specified in the assignment
        """
        weight = self.weight * self.connectivity
        strio = io.StringIO()
        for i in range(self.dim_in, self.dim_node):
            if i<self.dim_in+self.dim_hid and not self.hidden[i-self.dim_in]:   # no such hidden node
                continue
            strio.write('{:6.1f}'.format(weight[i][0]))
            for j in range(1, self.dim_node-1):
                if self.dim_in<=j<self.dim_in+self.dim_hid and not self.hidden[j-self.dim_in]:  # this node is not connected
                    continue
                strio.write(' {:6.1f}'.format(weight[i][j]))
            if i < self.dim_node - 1:
                strio.write('\n')
        return strio.getvalue()


    def __repr__(self):
        """ Return the whole weight matrix string (even not connected)
        """
        weight = self.weight * self.connectivity
        reprio = io.StringIO()
        remain_lines = self.dim_node
        for row in map(iter, weight):
            reprio.write('{:6.1f}'.format(next(row)))
            for ele in row:
                reprio.write(' {:6.1f}'.format(ele))
            if remain_lines > 1:
                reprio.write('\n')
                remain_lines -= 1
        return reprio.getvalue()


    def initialize(self, num_hid, dense, w_range, seed=None):
        """ Initialize the ANN according to the rule specified in the paper.

        :param num_hid: The initial number of hidden nodes
        :type num_hid: int
        :param dense: The initial connection density
        :type dense: float
        :param w_range: The inital connection weight range, in (-w_range, w_range)
        :type w_range: float
        :param seed: The random seed (for debugging purpose)
        :type seed: int
        """
        if not (0 < num_hid <= self.dim_hid):
            raise self.ANNException('hidden nodes should be within (0,{}]'.format(self.dim_hid))
        if not (0 < dense <= 1):
            raise self.ANNException('initial weight density should be within (0,1)')
        if w_range <= 0:
            raise self.ANNException('weight range should be positive')
        if seed is not None:
            np.random.seed(seed)

        self.hidden[:num_hid] = True

        # first make it fully connected
        for i in range(self.dim_in, self.dim_node):
            self.connectivity[i,:i] = True
        for i, has_hidden in enumerate(self.hidden, start=self.dim_in):
            if not has_hidden:
                self.connectivity[i,:] = False
                self.connectivity[:,i] = False
        # then flip some connection with probability
        for row in self.connectivity:
            for j, has_con in enumerate(row):
                if has_con and np.random.rand()>dense:
                    row[j] = False

        self.weight[:] = np.random.uniform(-w_range, w_range, self.weight.shape)
        self.weight *= self.connectivity


    def construct(self, weights):
        """ Construct the ANN from output weights specified by the assignment.

        :param weights: a weight matrix in the format specified by the assignment
        :type weights: np.ndarray

        :return: None
        :rtype: None
        """
        in_weights = weights    # first to append zero column as the last output (no out-degree)
        weights = np.zeros((weights.shape[0], weights.shape[1]+1))
        weights[:,:-1] = in_weights

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


    def _forward(self, x):
        """ Forward a vector to get the outputs of all nodes

        :param x: the vector represents all input nodes
        :type x: np.ndarray
        :return: the nodes output
        :rtype: np.ndarray
        """
        bias = -np.ones(1)
        tail = np.zeros(self.dim_hid+self.dim_out)
        nodes = np.concatenate((bias, x, tail), axis=0)
        weight = self.weight * self.connectivity
        for i in range(self.dim_in, self.dim_in+self.dim_hid+self.dim_out):
            net = nodes.dot(weight[i])
            nodes[i] = self.__sigmoid(net)

        nodes[self.dim_in:self.dim_in+self.dim_hid] *= self.hidden
        return nodes


    def _backpropagate(self, expected, nodes, lr):
        """ Back-propagating algorithm, for updating weights

        :param expected: the expected output vector
        :type expected: np.ndarray
        :param nodes: the output of all nodes in ANN
        :type nodes: np.ndarray
        :param lr: the learning rate
        :type lr: float
        """
        if not (0 < lr < 1):
            raise self.ANNException('learning rate cannot be negative or exceeds 1')

        gamma = np.zeros(nodes.shape)   # for calculating delta = gamma * sigmoid'(x)
        delta = np.zeros(nodes.shape)   # the delta variable (F_net in the book)

        gamma[-self.dim_out:] = nodes[-self.dim_out:] - expected    # delta of output nodes
        delta[-1] = gamma[-1]
        for i in range(self.dim_node-1, self.dim_in-1, -1):
            if i+1 < self.dim_node: # last node has no backpropagation
                gamma[i] += delta[i+1:].dot(self.weight[i+1:,i])
            delta[i] = self.__sigmoid(nodes[i], True) * gamma[i]

        self.weight -= lr * np.outer(delta, nodes)  # update weights


    def train(self):
        pass


    def evaluate(self, x):
        """ Calculate the output of ANN for input x.

        :param x: the input vector
        :type x: np.ndarry
        :return: the value of evaluated ANN
        :rtype: np.ndarry
        """
        if x.shape != (self.dim_in-1,):
            raise self.ANNException('input dimension not matching')
        nodes = self._forward(x)
        return nodes[-self.dim_out:]


    def test(self):
        pn = ParityNGenerator(7)
        vec = pn(22)
        exp = np.array([vec.sum() % 2 == 0])
        for i in range(1000):
            nodes = self._forward(vec)
            self._backpropagate(exp, nodes, 0.2)








    # TODO: four structural mutations
    def node_deletion(self):
        pass
    def connect_deletion(self):
        pass
    def node_addition(self):
        pass
    def connect_addition(self):
        pass







class SimulatedAnnealingSolver(object):
    def __init__(self):
        pass



class EvolutionaryProgramming(object):
    def __init__(self):
        pass

class EPNet(EvolutionaryProgramming):
    def __init__(self):
        super().__init__()


class NParityProblem(object):
    def __init__(self):
        pass



def main():
    parser = argparse.ArgumentParser(description='An N-parity problem solver based on evolutionary ANN')
    parser.add_argument('-s', type=int, required=True, help='An integer random seed', metavar='SEED', dest='seed')
    parser.add_argument('-n', default=5, type=int, help='The parameter N of N-parity problem', metavar='N', dest='n')
    parser.add_argument('-d', default=0.75, type=float, help='The initial connection density of forward ANN', metavar='DENSE', dest='dense')
    parser.add_argument('-r', default=10.0, type=float, help='The range of initial weights, from -R to R', metavar='R', dest='range')
    args = parser.parse_args()

    np.random.seed(args.seed)
    n = args.n


if __name__ == '__main__':
    main()

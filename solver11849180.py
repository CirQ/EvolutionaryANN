#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: cirq
# Created Time: 2018-09-23 18:07:29

import argparse
import functools
import heapq
import io
import queue
import threading

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
        """ Get a binary representation of integer num, and the answer of N-Parity
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
        return vec_num, vec_num.sum()%2==0

    def all(self):
        """ Return all binary vectors within self.bound

        :return: iterable (num, ans, vec) pairs
        :rtype: generator
        """
        for num in range(self.bound):
            vec, exp = self(num)
            yield num, np.array([exp]), vec



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


    def initialize(self, num_hid, dense, w_range=1.0, mean=0.0, stddev=1.0, seed=None):
        """ Initialize the ANN according to the rule specified in the paper.

        :param num_hid: The initial number of hidden nodes
        :type num_hid: int
        :param dense: The initial connection density
        :type dense: float
        :param mean: The mean value of a normal distribution
        :type mean: float
        :param stddev: The standard deviation of a normal distribution
        :type stddev: float
        :param seed: The random seed (for debugging purpose)
        :type seed: int
        """
        if not (0 < num_hid <= self.dim_hid):
            raise self.ANNException('hidden nodes should be within (0,{}]'.format(self.dim_hid))
        if not (0 < dense <= 1):
            raise self.ANNException('initial weight density should be within (0,1)')
        # if w_range <= 0:
        #     raise self.ANNException('weight range should be positive')
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

        # self.weight[:,:] = np.random.uniform(-w_range, w_range, self.weight.shape)
        self.weight[:,:] = np.random.normal(mean, stddev, self.weight.shape)
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
        bias = -np.ones((x.shape[0], 1))
        tail = np.zeros((x.shape[0], self.dim_hid+self.dim_out))
        nodes = np.concatenate((bias, x, tail), axis=1)
        weight = self.weight * self.connectivity
        for i in range(self.dim_in, self.dim_in+self.dim_hid+self.dim_out):
            net = nodes.dot(weight[i])
            nodes[:,i] = self.__sigmoid(net)
        nodes[:,self.dim_in:self.dim_in+self.dim_hid] *= self.hidden
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
        gamma = np.zeros(nodes.shape)   # for calculating delta = gamma * sigmoid'(x)
        delta = np.zeros(nodes.shape)   # the delta variable (F_net in the book)
        gamma[:,-self.dim_out:] = nodes[:,-self.dim_out:] - expected    # delta of output nodes
        delta[:,-1] = gamma[:,-1]
        for i in range(self.dim_node-1, self.dim_in-1, -1):
            if i+1 < self.dim_node: # last node has no backpropagation
                gamma[:,i] += delta[:,i+1:].dot(self.weight[i+1:,i])
            delta[:,i] = self.__sigmoid(nodes[:,i], True) * gamma[:,i]
        self.weight -= lr * np.matmul(delta.transpose(), nodes) # update weights


    def train(self, X, y, lr=0.3, epoch=100):
        """ Train the network with back propagation (fixed learning rate)

        :param X: the input matrix (or vector)
        :type X: np.ndarray
        :param y: the actual value of output
        :type y: np.ndarray
        :param lr: the learning rate
        :type lr: float
        :param epoch: the epochs of training
        :type epoch: int
        """
        if len(y.shape) == 1:
            y = y.reshape((-1, 1))
        if not (0 < lr < 1):
            raise self.ANNException('learning rate cannot be negative or exceeds 1')
        if epoch <= 0:
            raise self.ANNException('epoch must be postitive integer')
        for _ in range(epoch):
            nodes = self._forward(X)
            self._backpropagate(y, nodes, lr)


    def evaluate(self, x):
        """ Calculate the output of ANN for input x.

        :param x: the input vector
        :type x: np.ndarry
        :return: the value of evaluated ANN
        :rtype: np.ndarry
        """
        if len(x.shape) == 1:
            x = x.reshape((1, -1))
        if x.shape[1] != self.dim_in-1:
            raise self.ANNException('input dimension not matching')
        nodes = self._forward(x)
        return nodes[:,-self.dim_out:]


    def copy(self):
        """ Make a hard copy of this ANN

        :return: new copy of this ANN
        :rtype: ForwardArtificialNeuralNectwork
        """
        new_ann = ForwardArtificialNeuralNectwork(self.dim_in-1, self.dim_hid, self.dim_out)
        new_ann.weight[:,:] = self.weight
        new_ann.connectivity[:,:] = self.connectivity
        new_ann.hidden[:] = self.hidden
        return new_ann


    def _energy(self, X, y):
        """ Calculate the energy of this ANN

        :param X: the input matrix
        :type X: np.ndarray
        :param y: the corresponding result (supervised)
        :type y: np.ndarray
        :return: the energy of current ANN
        :rtype: float
        """
        yhat = self.evaluate(X)
        loss = ((y - yhat) ** 2).sum() / 2
        return loss


    def _to_neighbor(self, mean, stddev):
        """ Move the current network to its neighbor, and return the move

        :param mean: mean of a normal distribution
        :type mean: float
        :param stddev: standard deviation of a normal distribution
        :type stddev: float
        :return: the moves to the neighbor
        :rtype: np.ndarray
        """
        move = np.random.normal(mean, stddev, self.weight.shape)
        move *= self.connectivity
        self.weight += move
        return move


    cooldown_method = {
        'linear': lambda t, k: max(t - 0.005, 0.001),  # TODO: new cooldown
        'exponential': lambda t, k: 0.99 * t,
    }
    def simul_anneal(self, X, y, max_steps, temperature=1, cooldown='exponential', mean=0.0, stddev=1.0):
        cooldown = self.cooldown_method[cooldown]
        for i in range(max_steps):
            temperature = cooldown(temperature, i)
            before_energy = self._energy(X, y)
            move = self._to_neighbor(mean, stddev)
            after_energy = self._energy(X, y)
            dE = after_energy - before_energy
            if dE < 0.0 or np.exp(-dE/temperature) > np.random.rand():
                # accept the new state
                pass
            else:
                self.weight -= move


    # TODO: four structural mutations
    def node_deletion(self):
        pass
    def connect_deletion(self):
        pass
    def node_addition(self):
        pass
    def connect_addition(self):
        pass



class PriorityQueue(queue.PriorityQueue):
    """ Class of a priority queue, seperate priority and item. Also is a min heap
    """
    class PQueueException(Exception):
        pass


    def __init__(self, max_size):
        """ The constructor of Priority Queue

        :param max_size: the max_size of queue, but may exceed in runtime
        :type max_size: int
        """
        # max_size is different to maxsize from the superclass, the queue is
        # initialized to be infinite (by setting maxsize to 9), but will shrink
        # to max_size whenever self.constraint is called
        if max_size <= 0:
            raise self.PQueueException('max_size must be positive')
        super().__init__(maxsize=0)
        self.max_size = max_size

        # just like the mutex initialization in the Queue's constructor,
        # ref: https://github.com/python/cpython/blob/3.7/Lib/queue.py#L37
        self.mutating = threading.Condition(self.mutex)


    def put(self, item, priority=None, *args, **kwargs):
        """ the same as put except that priority must be specified
        """
        if priority is None:
            raise self.PQueueException('priority must be specified')
        super().put((priority, item), *args, **kwargs)


    def get(self, *args, **kwargs):
        """ the same as get except that priority and item is seperated
        """
        priority, item = super().get(*args, **kwargs)
        return priority, item


    def constraint(self):
        with self.mutating:
            self.queue = heapq.nsmallest(self.max_size, self.queue)
            heapq.heapify(self.queue)



class EPNet(object):
    def __init__(self, population_size):
        pass



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

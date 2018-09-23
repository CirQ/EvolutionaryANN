#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: cirq
# Created Time: 2018-09-23 18:07:29

import argparse

import numpy as np



def main():
    parser = argparse.ArgumentParser(description='An N-parity problem solver based on evolutionary ANN')
    parser.add_argument('-s', type=int, required=True, help='An integer random seed', metavar='SEED', dest='seed')
    parser.add_argument('-n', default=5, type=int, help='The parameter N of N-parity problem', metavar='N', dest='n')
    args = parser.parse_args()
    seed = args.seed
    n = args.n
    print(seed, n)


if __name__ == '__main__':
    main()

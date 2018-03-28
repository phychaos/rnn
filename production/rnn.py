#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Author: 林利芳
# @File: rnn.py
# @Time: 18-3-28 下午3:20

import numpy as np


class RecurrentNeuralNetwork(object):
    def __init__(self, word_dim=10, hidden_dim=100, output_dim=10):
        self.u = np.random.uniform(-1, 1, (hidden_dim, word_dim))
        self.w = np.random.uniform(-1, 1, (hidden_dim, hidden_dim))
        self.v = np.random.uniform(-1, 1, (hidden_dim, output_dim))

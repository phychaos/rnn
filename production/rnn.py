#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Author: 林利芳
# @File: rnn.py
# @Time: 18-3-28 下午3:20

import numpy as np


class RecurrentNeuralNetwork(object):
    def __init__(self, word_dim=10, hidden_dim=100, output_dim=10, batch=64, t_dim=10):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.batch = batch
        self.u = np.random.uniform(-1, 1, (hidden_dim, word_dim))
        self.w = np.random.uniform(-1, 1, (hidden_dim, hidden_dim))
        self.v = np.random.uniform(-1, 1, (output_dim, hidden_dim))
        self.s = np.zeros((t_dim, hidden_dim))
        self.o = np.zeros((t_dim, word_dim))

    def forward_propagation(self, x):
        s = np.zeros((len(x), self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        o = np.zeros((len(x), self.hidden_dim))
        for t, x_t in enumerate(x):
            s[t] = np.tanh(self.u.dot(x_t) + self.w.dot(self.s[t - 1]))
            o[t] = self.softmax(self.v.dot(self.s[t]))
        return o, s

    def softmax(self, x):
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    def predict(self, x):
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis=1)

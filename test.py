#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: Linlifang
# @file: test.py
# @time: 18-4-18下午5:14
from production.rnn import RecurrentNeuralNetwork
from core.utils import *

x_test, y_test = generate_test(5, dim=10)

rnn = RecurrentNeuralNetwork(word_dim=10, output_dim=10)


x_train, y_train = generate_data(n=5000, dim=10)
rnn.fit(x_train, y_train, lr=0.01)
rnn.test(x_test, y_test)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Author: 林利芳
# @File: rnn.py
# @Time: 18-3-28 下午3:20

import numpy as np


class RecurrentNeuralNetwork(object):
    def __init__(self, word_dim=10, hidden_dim=100, output_dim=10, batch=64):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.batch = batch
        self.input_w = np.random.random([hidden_dim, word_dim]) * 0.01
        self.input_b = np.random.random([hidden_dim]) * 0.01
        self.hidden_w = np.random.random([hidden_dim, hidden_dim]) * 0.01
        self.hidden_b = np.random.random([hidden_dim]) * 0.01
        self.output_w = np.random.random([output_dim, hidden_dim]) * 0.01
        self.output_b = np.random.random([output_dim]) * 0.01
        self.h = np.random.random([hidden_dim]) * 0.01

    def forward(self, x):
        """
        前向算法 计算隐藏层 预测结果
        :param x:
        :return:
        """
        s = np.zeros((len(x), self.hidden_dim))
        s[-1] = self.h
        o = np.zeros((len(x), self.word_dim))
        T = x.shape[1]
        for t in range(T):
            x_t = x[:, t]
            s[t] = np.tanh(np.dot(self.input_w, x_t) + self.input_b + np.dot(self.hidden_w, s[t - 1]))
            o[t] = self.softmax(np.dot(self.output_w, s[t]) + self.output_b)
        return o, s

    def backward(self, x, y, o, s, lr):
        """
        梯度下降法
        :param x: 训练集
        :param y: 训练集标签
        :param o: 预测标签
        :param s: 隐藏层
        :param lr: 学习率
        :return:
        """
        delta_wo = np.zeros_like(self.output_w)
        delta_bo = np.zeros_like(self.output_b)
        delta_wi = np.zeros_like(self.input_w)
        delta_bi = np.zeros_like(self.input_b)
        delta_wh = np.zeros_like(self.hidden_w)
        delta_bh = np.zeros_like(self.hidden_b)
        m, n = x.shape
        delta_ht = np.dot(np.transpose(self.output_w), o[-1] - y[:, -1])

        for t in range(m - 2, -1, -1):
            delta = (o[t] - y[:, t])
            dy = (1 - s[t] * s[t])
            delta_ht += np.dot(np.transpose(self.output_w), delta)
            delta_wo += np.outer(delta, s[t])
            delta_bo += delta
            delta_wi += np.outer(dy * delta_ht, x[:, t])
            delta_bi += dy * delta_ht

            delta_wh += np.outer(dy * delta_ht, s[t - 1])
            delta_bh += dy * delta_ht

        for param in [delta_wo, delta_bo, delta_wi, delta_bi, delta_wh, delta_bh]:
            np.clip(param, -5, 5, out=param)

        self.output_w -= lr * delta_wo / np.sqrt(delta_wo * delta_wo + 0.000000001)
        self.output_b -= lr * delta_bo / np.sqrt(delta_bo * delta_bo + 0.000000001)
        self.hidden_w -= lr * delta_wh / np.sqrt(delta_wh * delta_wh + 0.000000001)
        self.hidden_b -= lr * delta_bh / np.sqrt(delta_bh * delta_bh + 0.000000001)
        self.input_w -= lr * delta_wi / np.sqrt(delta_wi * delta_wi + 0.000000001)
        self.input_b -= lr * delta_bi / np.sqrt(delta_bi * delta_bi + 0.000000001)
        self.h -= lr * delta_ht / np.sqrt(delta_ht * delta_ht + 0.000000001)

    def predict(self, x):
        o = np.zeros((len(x), self.word_dim))
        ht = self.h
        T = x.shape[1]
        for t in range(T):
            x_t = x[:, t]
            ht = np.tanh(np.dot(self.input_w, x_t) + self.input_b + np.dot(self.hidden_w, ht))
            o[t] = self.softmax(np.dot(self.output_w, ht) + self.output_b)
        return o

    def softmax(self, x):
        exp_x = np.exp(x) + 0.00000000001
        return exp_x / np.sum(np.exp(x) + 0.00000000001)

    def fit(self, x_train, y_train, lr=0.01):
        num = x_train.shape[0]
        for kk in range(num):
            x, y = x_train[kk], y_train[kk]
            o, s = self.forward(x)
            self.backward(x, y, o, s, lr)

    def test(self, x, y):
        o = self.predict(x)
        result = [[], [], []]
        for ii, jj, kk in zip(x.T, y.T, o):
            result[0].append(np.argmax(ii))
            result[1].append(np.argmax(jj))
            result[2].append(np.argmax(kk))
        print('样本特征:\t{}'.format(result[0]))
        print('样本标签:\t{}'.format(result[1]))
        print('样本预测:\t{}'.format(result[2]))

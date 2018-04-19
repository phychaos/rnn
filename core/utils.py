#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: Linlifang
# @file: utils.py
# @time: 18-4-19上午9:54
import numpy as np


def generate_data(n=1000, dim=10):
    """
    产生数据集
    :param n: 训练集数据量
    :param dim: 维度
    :return:
    """
    x_train = np.zeros((n, dim, dim))
    y_train = np.zeros((n, dim, dim))
    for kk in range(n):
        start_num = np.random.randint(0, dim)
        next_num = start_num + 1
        _x = []
        _y = []
        for i in range(dim):
            if start_num == dim:
                start_num = 0
            if next_num == dim:
                next_num = 0
            x_train[kk, start_num, i] = 1
            y_train[kk, next_num, i] = 1

            x = np.argmax(x_train[kk, i])
            y = np.argmax(y_train[kk, i])
            _x.append(x)
            _y.append(y)
            start_num += 1
            next_num += 1
    return x_train, y_train


def generate_test(start_num=5, dim=10):
    """
    产生测试集
    :param start_num: 起始数据
    :param dim: 维度
    :return:
    """
    x = np.zeros((dim, dim))
    y = np.zeros((dim, dim))
    next_num = start_num + 1
    for i in range(dim):
        if start_num == dim:
            start_num = 0
        if next_num == dim:
            next_num = 0
        x[start_num, i] = 1
        y[next_num, i] = 1
        start_num += 1
        next_num += 1
    return x, y

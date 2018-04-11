#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def loadData():
    dataMat = []
    labelMat = []
    with open("testSet.txt", "r") as f:
        for line in f.readlines():
            lineArr = line.strip().split()
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
            labelMat.append(int(lineArr[2]))
            # print(dataMat, labelMat)
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


def gradAscent(dataMat, labelMat):
    dataMatrix = np.mat(dataMat)
    labelMatrix = np.mat(labelMat).transpose()
    n, m = np.shape(dataMatrix)
    alpha = 0.1
    maxCycles = 500
    weights = np.ones((m, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMatrix - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


def drawBestFit(weights):
    weights = np.array(weights)
    data, label = loadData()
    dataArray = np.array(data)
    x = dataArray[:, 1]
    y = dataArray[:, 2]
    l = np.array(label)
    plt.scatter(x, y, s=50 * (l + 0.2), c=l)
    plt.subplot(111)
    step = np.arange(-3, 3, 0.1)
    val = (-weights[0] - weights[1] * step) / weights[2]
    plt.plot(step, val)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()


def stocGradAscent(dataMatrix, classLables, num):
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    weightsList = []
    for j in range(num):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (i + j + 1) + 0.01
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLables[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            weightsList.append(weights.tolist())
            del (dataIndex[randIndex])
    plotWeights(weightsList)
    return weights


def plotWeights(weightsList):
    axis = list(range(len(weightsList)))
    plt.subplot(311)
    plt.title(" weight 0")
    print(weightsList)
    plt.plot(axis, [x[0] for x in weightsList], c="#054E9F")
    plt.subplot(312)
    plt.title(" weight 1")
    plt.plot(axis, [x[1] for x in weightsList], c="red")
    plt.subplot(313)
    plt.title(" weight 2")
    plt.plot(axis, [x[2] for x in weightsList], c="yellow")
    plt.show()


if __name__ == '__main__':
    data, label = loadData()
    matrix = gradAscent(data, label)
    print(matrix)
    drawBestFit(matrix)
    stocMatrix = stocGradAscent(np.array(data), label, 80)
    print(stocMatrix)
    drawBestFit(stocMatrix)

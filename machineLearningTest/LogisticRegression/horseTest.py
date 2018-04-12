#!/usr/bin/python
# -*- coding: UTF-8 -*-
from lr import *


def classfy(inX, weights):
    prop = sigmoid(sum(inX * weights))
    return 1 if (prop > 0.5) else 0


def colicTest():
    trainLabel, trainSet = loadTrainData()
    trainWeights = stocGradAscent(np.array(trainSet), trainLabel, 600)
    errorRate = classfyTestData(trainWeights)
    return errorRate


def classfyTestData(trainWeights):
    errorCount = 0
    numTestVec = 0.00
    with open("horseColicTest.txt", "r") as frTest:
        for line in frTest.readlines():
            numTestVec += 1.0
            currLine = line.strip().split()
            lineArr = [0.0]
            for i in range(21):
                lineArr.append(float(currLine[i]))
            if int(classfy(np.array(lineArr), trainWeights) != int(currLine[21])):
                errorCount += 1
    errorRate = float(errorCount) / numTestVec
    print(" error rate is : " + str(errorRate))
    return errorRate


def loadTrainData():
    trainSet = []
    trainLabel = []
    with open("horseColicTraining.txt", "r") as frTrain:
        for line in frTrain.readlines():
            currLine = line.strip().split()
            lineArr = []
            lineArr.append(0.0)
            for i in range(21):
                lineArr.append(float(currLine[i]))
            trainSet.append(lineArr)
            trainLabel.append(float(currLine[21]))
    return trainLabel, trainSet


def multiTest(num):
    errorSum = 0.0
    for k in range(num):
        errorSum += colicTest()
    print("after %d interations the average error is %f " % (num, errorSum / float(num)))


if __name__ == '__main__':
    multiTest(10)

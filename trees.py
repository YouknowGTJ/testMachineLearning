#!/usr/bin/python
# -*- coding: UTF-8 -*-

import math
import operator


def calShannonEnt(dataSet):
    num = len(dataSet)
    labelCount = {}
    for featVec in dataSet:
        # print(featVec)
        currentLabel = featVec[-1]
        # print(featVec[-1])  # the last element in list .
        if currentLabel not in labelCount.keys():
            labelCount[currentLabel] = 0
        labelCount[currentLabel] += 1
    # calculate shannon ent
    shannonEnt = 0
    for key, val in labelCount.items():
        print(key, val)
        prop = float(val) / num
        shannonEnt -= prop * math.log2(prop)
    return shannonEnt


def createDataSet():
    dataSet = [[1, 1, "yes"],
               [1, 1, "yes"],
               [1, 0, "no"],
               [0, 1, "no"],
               [0, 1, "no"]]
    labels = ['no surfacing', 'flippers']
    # change to discrete values
    return dataSet, labels


# 根据指定的特征以及该特征的值划分数据集
def splitDataSet(dataSet, axis, value):
    resultSet = []
    for item in dataSet:
        if item[axis] == value:
            temp = item[:axis]
            temp.extend(item[axis:])
            resultSet.append(temp)
    return resultSet


def chooseBestFeature(dataSet):
    featureNum = len(dataSet[0]) - 1
    baseEntropy = calShannonEnt(dataSet)
    bestFeature = -1
    gain = 0.0
    for i in range(featureNum):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)  # 对于每个特征， 不同取值的集合
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calShannonEnt(subDataSet)  # 对于划分的数据子集，分别求信息熵，再加权求和
        if (baseEntropy - newEntropy > gain):
            gain = baseEntropy - newEntropy
            bestFeature = i
    return bestFeature, gain


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1:  # stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeature(dataSet)[0]
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    # del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]  # copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)  # 此处划分了数据集
    return myTree


if __name__ == '__main__':
    data, labels = createDataSet()
    shannoEnt = calShannonEnt(data)
    print(shannoEnt)
    print(splitDataSet(data, 0, 1))
    print(splitDataSet(data, 0, 0))
    print(chooseBestFeature(data))
    print(createTree(data, labels))

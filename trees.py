#!/usr/bin/python
# -*- coding: UTF-8 -*-

import  math

def calShannonEnt(dataSet):
    num = len(dataSet)
    labelCount = {}
    for featVec in dataSet:
        #print(featVec)
        currentLabel = featVec[-1]
        #print(featVec[-1])  # the last element in list .
        if currentLabel not in labelCount.keys():
            labelCount[currentLabel] = 0
        labelCount[currentLabel] += 1
    #calculate shannon ent
    shannonEnt = 0
    for key,val in labelCount.items():
        print(key , val)
        prop = float(val) / num
        shannonEnt -= prop * math.log2(prop)
    return shannonEnt

def createDataSet():
    dataSet = [[1, 1, "yes"],
               [1, 1, "yes"],
               [1, 0, "no"],
               [0, 1, "no"],
               [0, 1, "no"]]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

if __name__ == '__main__':
    data, labels = createDataSet()
    shannoEnt = calShannonEnt(data)
    print(shannoEnt)
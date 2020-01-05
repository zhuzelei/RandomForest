import pandas as pd
import numpy as np
import copy
import random
import math
import _thread
import time


# 初始化
trainSize = 5
testSize = 6
treeCounts = 20
treeList = []  # 训练所得树集合


# 主函数：读取数据并调用子树线程，完成训练后进行测试集测试
def main():

    X = pd.DataFrame()  # 初始训练data
    y = pd.DataFrame()  # 初始训练label
    # testHousing()

    # 读取文件
    print("开始读取训练集数据并训练模型")
    for i in range(trainSize):
        # 读取训练集
        # X = pd.concat(
        #     [X, pd.read_csv("data/train"+str(i+1)+".csv", header=None)], axis=0)
        # y = pd.concat(
        #     [y, pd.read_csv("data/label"+str(i+1)+".csv", header=None)], axis=0)
        print("     已读取训练集index:" + str(i+1))
    X = pd.read_csv("data/tmpTrain.csv", header=None)
    y = pd.read_csv("data/tmpLabel.csv", header=None)
    y.columns = ['label']
    df = pd.concat([X, y], axis=1)
    labels = df.columns.values.tolist()
    # print(df)

    # 开始训练
    for i in range(treeCounts):
        _thread.start_new_thread(trainThreadRun, (df,))

    while len(treeList) != treeCounts:
        pass

    # 此处只用测试数据进行预测
    labelPred = []
    for tree in treeList:
        testData = [-0.125,-44,20,7,75,-345,-23,0.84508,0.94717,0.91958,0.93313,0.90008,0.97021]
        label = test(tree, labels[:-1], testData)
        labelPred.append(label)
    print("The predicted value is: {}".format(np.mean(labelPred)))

# 子树线程：取样和训练
def trainThreadRun(df):
    baggingData, bagginglabels = bagging(df)
    tree = train(baggingData, bagginglabels)
    treeList.append(tree)
    print("子树训练完成: ", len(treeList))

# Bagging 抽取样本: sqrt(m-1)
def bagging(dataSet):
    n, m = dataSet.shape
    features = random.sample(
        list(dataSet.columns.values[:-1]), int(math.sqrt(m - 1)))
    features.append(dataSet.columns.values[-1])
    rows = [random.randint(0, n-1) for _ in range(n)]
    trainData = dataSet.iloc[rows][features]
    return trainData.values.tolist(), features

# 建立决策树
def train(dataSet, features):
    classList = [dt[-1] for dt in dataSet]
    # label一样，全部分到一边
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 最后一个特征还不能把所有样本分到一边，则划分到平均值
    if len(features) == 1:
        return np.mean(classList)
    bestFeatureIndex, bestSplitValue = getBestFeat(dataSet)
    bestFeature = features[bestFeatureIndex]
    # 删除root特征，生成新的去掉root特征的数据集
    newFeatures, leftData, rightData = splitData(
        dataSet, bestFeatureIndex, features, bestSplitValue)

    # 左右子树有一个为空，则返回该节点下样本均值
    if len(leftData) == 0 or len(rightData) == 0:
        return np.mean([dt[-1] for dt in leftData] + [dt[-1] for dt in rightData])
    else:
        # 左右子树不为空，则继续分裂
        myTree = {bestFeature: {
            '<' + str(bestSplitValue): {}, '>' + str(bestSplitValue): {}}}
        myTree[bestFeature]['<' +
                            str(bestSplitValue)] = train(leftData, newFeatures)
        myTree[bestFeature]['>' +
                            str(bestSplitValue)] = train(rightData, newFeatures)
    return myTree

# 根据选取的最优属性生成新的数据集（分裂为左右子树）
def splitData(dataSet, featIndex, features, value):
    newFeatures = copy.deepcopy(features)
    newFeatures.remove(features[featIndex])
    leftData, rightData = [], []
    for dt in dataSet:
        temp = []
        temp.extend(dt[:featIndex])
        temp.extend(dt[featIndex + 1:])
        if dt[featIndex] <= value:
            leftData.append(temp)
        else:
            rightData.append(temp)
    return newFeatures, leftData, rightData

# 获取当前结点数据集基尼指数最小的特征（最优属性获取）；返回对应特征index和对应划分值
def getBestFeat(dataSet):
    bestR2 = float('inf')
    bestFeatureIndex = -1
    bestSplitValue = None
    # 第i个特征
    for i in range(len(dataSet[0]) - 1):
        featList = [dt[i] for dt in dataSet]
        # 产生候选划分点
        sortfeatList = sorted(list(set(featList)))
        splitList = []
        # 如果值相同，不存在候选划分点
        if len(sortfeatList) == 1:
            splitList.append(sortfeatList[0])
        else:
            for j in range(len(sortfeatList) - 1):
                splitList.append((sortfeatList[j] + sortfeatList[j + 1]) / 2)
        # 第j个候选划分点，记录最佳划分点
        for splitValue in splitList:
            subDataSet0, subDataSet1 = splitDataSet(dataSet, i, splitValue)
            lenLeft, lenRight = len(subDataSet0), len(subDataSet1)
            # 防止数据集为空，mean不能计算
            if lenLeft == 0 and lenRight != 0:
                rightMean = np.mean(subDataSet1)
                R2 = sum([(x - rightMean)**2 for x in subDataSet1])
            elif lenLeft != 0 and lenRight == 0:
                leftMean = np.mean(subDataSet0)
                R2 = sum([(x - leftMean) ** 2 for x in subDataSet0])
            else:
                leftMean, rightMean = np.mean(
                    subDataSet0), np.mean(subDataSet1)
                leftR2 = sum([(x - leftMean)**2 for x in subDataSet0])
                rightR2 = sum([(x - rightMean)**2 for x in subDataSet1])
                R2 = leftR2 + rightR2
            if R2 < bestR2:
                bestR2 = R2
                bestFeatureIndex = i
                bestSplitValue = splitValue
    return bestFeatureIndex, bestSplitValue
    
# 对连续变量划分数据集，返回数据只包括最后一列
def splitDataSet(dataSet, featIndex, value):
    leftData, rightData = [], []
    for dt in dataSet:
        if dt[featIndex] <= value:
            leftData.append(dt[-1])
        else:
            rightData.append(dt[-1])
    return leftData, rightData


# 用生成的回归树对测试样本进行测试
def test(decisionTree, featureLabel, testDataSet):
    firstFeature = list(decisionTree.keys())[0]
    secondFeatDict = decisionTree[firstFeature]
    splitValue = float(list(secondFeatDict.keys())[0][1:])
    featureIndex = featureLabel.index(firstFeature)
    if testDataSet[featureIndex] <= splitValue:
        valueOfFeat = secondFeatDict['<' + str(splitValue)]
    else:
        valueOfFeat = secondFeatDict['>' + str(splitValue)]

    # 还没到叶结点需要继续下溯
    if isinstance(valueOfFeat, dict):
        pred_label = test(valueOfFeat, featureLabel, testDataSet)
    else:
        pred_label = valueOfFeat
    return pred_label


if __name__ == '__main__':
    main()

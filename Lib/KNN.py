import random
import numpy as np
from operator import itemgetter

from scipy.spatial import distance
import pandas as pd


class KNN:
    def __init__(self, data, test, k, classColumnName):
        self.data = data
        #self.data, self.test = self.splitTrainTest(10)
        self.data = data
        self.test = test
        self.k = k
        self.classColumnName = classColumnName

    @staticmethod
    def getDistEuclidean(vector1, vector2):
        return distance.euclidean(vector1, vector2)


    def splitTrainTest(self, testPercent):
        trainData = []
        testData = []
        for row in self.data:
            if random.random() < testPercent:
                testData.append(row)
            else:
                trainData.append(row)
        return trainData, testData

    @staticmethod
    def getAccurancy(resultDataFrame, actualResultColumnName, predictedResultColumnName):
        counter = 0
        length = len(resultDataFrame)
        for i in range(length):
            if resultDataFrame.iloc[i][actualResultColumnName] != resultDataFrame.iloc[i][predictedResultColumnName]:
                counter = counter+1
        return 1-(counter / length)

    def generalCalculations(self):
        y_data = self.data[self.classColumnName]
        data = self.data.drop([self.classColumnName], axis=1)
        kNeighboursForTest = []
        for test_index, test_row in self.test.iterrows():
            labels = []
            testDistances = {}
            for data_index , data_row in data.iterrows():
                testDistances[data_index] = self.getDistEuclidean(test_row, data_row)
            testDistancesSorted = (sorted(testDistances.items(), key=itemgetter(1)))

            for i in range(0, self.k):
                labels.append(y_data[testDistancesSorted[i][0]])
            kNeighboursForTest.append(labels)
        return kNeighboursForTest

    def Regression(self):
        kNeighboursForTest = self.generalCalculations()
        length = len(kNeighboursForTest)
        labeled_test = []
        for i in range(length):
            avg_val = np.mean(kNeighboursForTest[i])
            labeled_test.append(avg_val)
        return labeled_test

    def Classify(self):
        kNeighboursForTest = self.generalCalculations()
        length = len(kNeighboursForTest)
        labeled_test = []
        for i in range(length):
            a_set = set(kNeighboursForTest[i])

            most_common = None
            qty_most_common = 0

            for item in a_set:
                qty = kNeighboursForTest[i].count(item)
                if qty > qty_most_common:
                    qty_most_common = qty
                    most_common = item

            labeled_test.append(most_common)

        return pd.DataFrame(labeled_test)





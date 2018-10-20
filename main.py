from pandas import DataFrame, Series
import pandas as pd
from Lib import KNN

data = pd.read_csv("C:\\Users\\fungo\\PycharmProjects\\lab_2\\data\\Habberman.csv",
                   sep=';', names=['Age', 'year of operation', 'Number of positive axillary', 'Survival status'])
test = pd.read_csv("C:\\Users\\fungo\\PycharmProjects\\lab_2\\data\\Habberman.test.csv",
                   sep=';', names=['Age', 'year of operation', 'Number of positive axillary', 'Survival status'])

y_test = test['Survival status']
test = test.drop('Survival status', axis=1)
result_classify = DataFrame()
result_classify['was'] = y_test

knn_classifier = KNN.KNN(data, test, 5, 'Survival status')
result_classify.insert(1, 'predict', knn_classifier.Classify())
print('точность классификации: ', knn_classifier.getAccurancy(result_classify, 'was', 'predict'))
print(result_classify)

df = pd.read_csv("C:\\Users\\fungo\\PycharmProjects\\lab_2\\data\\dataset_Facebook.csv", sep=";",
                 names=['Page total likes', 'Category',
                        'Post Weekday', 'Post Hour', 'Paid',
                        'Lifetime Post Total Reach', 'Lifetime Post Total Impressions',
                        'Lifetime Engaged Users', 'Lifetime Post Consumers', 'Lifetime Post Consumptions',
                        'Lifetime Post Impressions by people who have liked your Page',
                        'Lifetime Post reach by people who like your Page',
                        'Lifetime People who have liked your Page and engaged with your post',
                        'Total Interactions'])
dfTest = pd.read_csv("C:\\Users\\fungo\\PycharmProjects\\lab_2\\data\\dataset_Facebook.csv", sep=";",
                     names=['Page total likes', 'Category',
                            'Post Weekday', 'Post Hour', 'Paid',
                            'Lifetime Post Total Reach', 'Lifetime Post Total Impressions',
                            'Lifetime Engaged Users', 'Lifetime Post Consumers', 'Lifetime Post Consumptions',
                            'Lifetime Post Impressions by people who have liked your Page',
                            'Lifetime Post reach by people who like your Page',
                            'Lifetime People who have liked your Page and engaged with your post',
                            'Total Interactions'])
X = df[:450]
X_test = dfTest[451:498]
Y = df.drop("Total Interactions", axis=1)[451:498]
Y_test = dfTest["Total Interactions"][451:498]

result_regress = DataFrame()
result_regress['was'] = Y_test
knn_regression = KNN.KNN(X, Y, 4, "Total Interactions")
result_regress.insert(1, 'predict', knn_regression.Regression())
print(result_regress)

import pandas

from Lib import Document
from Lib import PLSA

corp = []
file = pandas.read_csv("C:\\Users\\Fungo\\PycharmProjects\\lab_2\\data\\lenta_ru.csv")
handle = open("C:\\Users\\Fungo\\PycharmProjects\\lab_2\\data\\lenta_ru.csv", "r", encoding="utf-8")

for i in range(15):
    text = handle.readline()
    corp.append(Document.Document(text))

max_iterations = 10
plsa = PLSA.PLSA(corp, max_iterations)
plsa.train(10)


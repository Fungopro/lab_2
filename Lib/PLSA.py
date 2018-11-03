import numpy as np
from operator import itemgetter


def normalize(vec):
    s = sum(vec)
    assert (abs(s) != 0.0)  # the sum must not be 0

    for i in range(len(vec)):
        assert (vec[i] >= 0)  # element must be >= 0
        vec[i] = vec[i] * 1.0 / s


def build_vocabulary(corpus):
    """ получаем словарь терминов на основе всего корпуса """
    discrete_set = set()
    for document in corpus:
        for word in document.terms:
            discrete_set.add(word)
    return list(discrete_set)


class PLSA(object):

    def __init__(self, corpus, number_of_topics):
        self.n_d = len(corpus)
        self.vocabulary = build_vocabulary(corpus)
        self.n_w = len(self.vocabulary)
        self.n_t = number_of_topics
        self.L = 0.0
        self.corpus = corpus
        # bag of words
        self.n_w_d = np.zeros([self.n_d, self.n_w], dtype=np.int)
        for di, doc in enumerate(corpus):
            n_w_di = np.zeros([self.n_w], dtype=np.int)
            for word in doc.terms:
                if word in self.vocabulary:
                    word_index = self.vocabulary.index(word)
                    n_w_di[word_index] = n_w_di[word_index] + 1
            self.n_w_d[di] = n_w_di

        # P(z|w,d)
        self.p_z_dw = np.zeros([self.n_d, self.n_w, self.n_t], dtype=np.float)
        # P(z|d)
        self.p_z_d = np.random.random(size=[self.n_d, self.n_t])
        for di in range(self.n_d):
            normalize(self.p_z_d[di])
        # P(w|z)
        self.p_w_z = np.random.random(size=[self.n_t, self.n_w])
        for zi in range(self.n_t):
            normalize(self.p_w_z[zi])

    def log_likelihood(self):
        l = 0
        for di in range(self.n_d):
            for wi in range(self.n_w):
                sum1 = 0
                for zi in range(self.n_t):
                    sum1 = sum1 + self.p_z_d[di, zi] * self.p_w_z[zi, wi]
                l = l + self.n_w_d[di, wi] * np.log(sum1)
        return l

    def print_p_z_d(self):
        filename = "C:\\Users\\Fungo\\PycharmProjects\\lab_2\\results\\p_z_d.txt"
        f = open(filename, "w")
        for di in range(self.n_d):
            f.write("Doc #" + str(di) + ":")
            for zi in range(self.n_t):
                f.write(" " + str(self.p_z_d[di, zi]))
            f.write("\n")
        f.close()

    def print_top_words(self, topk):
        filename = "C:\\Users\\Fungo\\PycharmProjects\\lab_2\\results\\top_words.txt"
        f = open(filename, "w")
        for zi in range(self.n_t):
            word_prob = self.p_w_z[zi, :]
            word_index_prob = []
            for wi in range(self.n_w):
                word_index_prob.append([wi, word_prob[wi]])
            word_index_prob = sorted(word_index_prob, key=itemgetter(1), reverse=True)
            f.write("-------------\n" + "Topic #" + str(zi) + ":\n")
            for wi in range(topk):
                index = word_index_prob[wi][0]
                prob = word_index_prob[wi][1]
                f.write(self.vocabulary[index] + " " + str(prob) + "\n")
        f.close()

    def train(self, max_iter):

        print("Training...")

        for i_iter in range(max_iter):

            # likelihood
            self.L = self.log_likelihood()

            self.print_top_words(10)

            print("Iter " + str(i_iter) + ", L=" + str(self.L))

            # Шаг E
            for di in range(self.n_d):
                for wi in range(self.n_w):
                    sum_zk = np.zeros([self.n_t], dtype=float)
                    for zi in range(self.n_t):
                        sum_zk[zi] = self.p_z_d[di, zi] * self.p_w_z[zi, wi]
                    sum1 = np.sum(sum_zk)
                    if sum1 == 0:
                        sum1 = 1
                    for zi in range(self.n_t):
                        self.p_z_dw[di, wi, zi] = sum_zk[zi] / sum1

            # Шаг M
            # обновление матрицы P(z|d)
            for di in range(self.n_d):
                for zi in range(self.n_t):
                    sum1 = 0.0
                    sum2 = 0.0
                    for wi in range(self.n_w):
                        sum1 = sum1 + self.n_w_d[di, wi] * self.p_z_dw[di, wi, zi]
                        sum2 = sum2 + self.n_w_d[di, wi]
                    if sum2 == 0:
                        sum2 = 1
                    self.p_z_d[di, zi] = sum1 / sum2

            # обнолвление матирицы P(w|z)
            for zi in range(self.n_t):
                sum2 = np.zeros([self.n_w], dtype=np.float)
                for wi in range(self.n_w):
                    for di in range(self.n_d):
                        sum2[wi] = sum2[wi] + self.n_w_d[di, wi] * self.p_z_dw[di, wi, zi]
                sum1 = np.sum(sum2)
                if sum1 == 0:
                    sum1 = 1
                for wi in range(self.n_w):
                    self.p_w_z[zi, wi] = sum2[wi] / sum1

        self.print_p_z_d()

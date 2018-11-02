from nltk import ngrams
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
import nltk
stop_words = nltk.corpus.stopwords.words('russian')


class Document:

    def __init__(self, doc):
        self.terms = None
        self.doc_arr = None
        self.doc = doc

    def prepare(self):
        tokenizer = RegexpTokenizer(r'\w+')
        self.doc_arr = tokenizer.tokenize(self.doc)
        stemmer = SnowballStemmer("russian")
        self.doc_arr = [stemmer.stem(word) for word in self.doc_arr]
        temp = []
        for word in self.doc_arr:
            if word not in stop_words:
                temp.append(word)
        self.doc_arr = temp

    def get_term_list(self):
        self.prepare()

        token = " "
        token = token.join(self.doc_arr)
        bigrams = ngrams(token, 2)
        trigrams = ngrams(token, 3)

        for k1, k2 in nltk.Counter(bigrams):
            self.terms.append(k1 + "_" + k2)

        for k1, k2, k3 in nltk.Counter(trigrams):
            self.terms.append(k1 + "_" + k2 + "_" + k3)

        return self.terms

from nltk.util import ngrams
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
import nltk

stop_words = nltk.corpus.stopwords.words('russian')


def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])


class Document:

    def __init__(self, doc):
        self.doc = doc
        self.doc_arr = None
        self.terms = self.get_term_list()

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
        terms = []
        bigrams = find_ngrams(self.doc_arr, 2)
        trigrams = find_ngrams(self.doc_arr, 3)
        print(self.doc)
        print(self.doc_arr)
        for words in bigrams:
            terms.append(words[0]+" "+words[1])

        for words in trigrams:
            terms.append(words[0]+" "+words[1]+" "+words[2])

        return terms

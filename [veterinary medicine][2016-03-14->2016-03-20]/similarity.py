from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from gensim import corpora, models, similarities

class similar(object):

    def __init__(self, texts):
        self.vecs = []
        self.text_lsi = self.train_lsi_model(texts, 5)

    def train_lsi_model(self, texts, num_of_toptics=10):
        texts_tokenized = [[word.lower()
                          for word in word_tokenize(text)]
                          for text in texts]
        # remove the stop words and punctuations
        english_stop_words = stopwords.words('english')
        english_punctuations = [',', '.', ':', '?', '(', ')', '[',
                                ']', '@', '&', '!', '*', '#', '$', '%']
        texts_filtered = [[word for word in text_tokenized
                         if (not word in english_punctuations) and
                         (not word in english_stop_words)]
                         for text_tokenized in texts_tokenized]
        # stem the word
        st = LancasterStemmer()
        texts_stemed = [[st.stem(word) for word in text_filtered]
                       for text_filtered in texts_filtered]

        all_stems = sum(texts_stemed, [])
        stem_once = set(stem for stem in set(all_stems)
                        if all_stems.count(stem) == 1)
        cleaned_texts = [[stem for stem in text if stem not in stem_once]
                        for text in texts_stemed]

        dictionary = corpora.Dictionary(cleaned_texts)
        corpus = [dictionary.doc2bow(text) for text in cleaned_texts]
        tfidf = models.TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]
        lsi = models.LsiModel(corpus_tfidf, id2word=dictionary,
                              num_topics=num_of_toptics)
        result = lsi[corpus]
        return result

    def get_vecs(self):
        vecs = [[i[1] for i in lsi] for lsi in self.text_lsi]
        return vecs

    def distance(self, vec1, vec2):
        lsi1 = [(i, vec1[i]) for i in range(len(vec1))]
        lsi2 = [(i, vec2[i]) for i in range(len(vec2))]
        index = similarities.MatrixSimilarity([lsi1, lsi2])
        sim = abs(index[lsi1][1])
        # print('[INFO]',sim)
        return sim

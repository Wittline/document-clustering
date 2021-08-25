import pandas as pd
import re
import numpy as np
import nltk
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from scipy.cluster.hierarchy import ward, dendrogram
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

class AHC(object):

    def __init__(self, filename, columns, t_column, d_column):
        self.filename = filename
        self.columns = columns
        self.title_column = t_column
        self.description_column = d_column
        self.df = None

    def process(self, show=True):
        self.df = pd.read_csv(self.filename)
        self.df = self.df[self.columns]
        self.df[self.description_column].fillna('', inplace=True)
        self.df[self.description_column] = self.df[self.title_column] + '. ' +  self.df[self.description_column].map(str)
        self.df.dropna(inplace=True)
        self.df.drop_duplicates(inplace=True)
        return self.df

    def __normalize(self, d):
        stopwords = nltk.corpus.stopwords.words('english')
        d = re.sub(r'[^a-zA-Z0-9\s]', '', d, re.I|re.A)
        d = d.lower().strip()
        tks = nltk.word_tokenize(d)
        f_tks = [t for t in tks if t not in stopwords]
        return ' '.join(f_tks)

    def get_normalized_corpus(self, tokens = False):
        n_corpus = np.vectorize(self.__normalize)        
        if tokens == True:
            norm_courpus = n_corpus(list(self.df[self.description_column]))
            return np.array([nltk.word_tokenize(d) for d in norm_corpus])            
        else:
            return n_corpus(list(self.df[self.description_column]))        

    def hierarchical_clustering(self, m):
        cd = 1 - cosine_similarity(m)        
        return ward(cd)

    def plot(self, lm, movies, p, figure_size, title_name_column):

        fig, ax = plt.subplots(figsize=figure_size)
        titles = movies[title_name_column].values.tolist()
        r = dendrogram(lm, orientation='left', labels = titles,
                       truncate_mode= 'lastp',
                       p=p,
                       no_plot=True)
        
        t = {r['leaves'][i]: titles[i] for i in range(len(r['leaves']))}

        def lab(x):
            return '{}'.format(t[x])
        
        ax = dendrogram(
            lm,
            truncate_mode = 'lastp',
            orientation = 'left',
            p = p,
            leaf_label_func = lab,
            leaf_font_size = 7.,
        )

        plt.tick_params(axis='x',
                        which= 'both',
                        bottom = 'off',
                        top = 'off',
                        labelbottom = 'off')
        
        plt.tight_layout()
        plt.savefig('hierarchical.png', dpi=200)


# mr = AHC('archive.zip', ['Movie Name', 'Plot'], 'Movie Name', 'Plot')
# df = mr.process()

# norm_corpus = mr.get_normalized_corpus()

# stopwords = nltk.corpus.stopwords.words('english')
# cv = CountVectorizer(ngram_range =(1,2), min_df = 10, max_df = 0.8, stop_words= stopwords)
# mat = cv.fit_transform(norm_corpus)
# mat.shape


# lm = mr.hierarchical_clustering(mat)
# title_name_column = 'Movie Name'
# mr.plot(lm, df, 1000, (15, 80), title_name_column)
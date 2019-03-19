
# -*- coding: utf-8 -*-

"""
Play with alpha!
"""

import pandas as pd
import numpy as np
import re
import random
import sqlite3


class MyLDA:
    
    pwd = './'
    n_docs = 10
    sample = False
    corpus = pwd + '/poetics.txt'
    n_topics = 5
    n_iters = 100
    alpha = 0.01
    beta = .1
    n_topwords = 5
    show_topics_interval = 5
    db_file = pwd + '/lda2.db'
    
    stopwords_file = pwd + '/stopwords.txt'
    LETTERS = re.compile(r"[^a-z]")

    def __init__(self):
        self.db = sqlite3.connect(self.db_file)
        
    def __del__(self):
        self.db.close()
        
    def save(self, df, table_name, if_exists='replace', index=True):
        df.to_sql(table_name, self.db, if_exists=if_exists, index=index)
        
    def make_corpus(self):
        
        # Import docs file into doc table
        docs = pd.read_csv(self.corpus)
        if self.sample: self.docs = docs.sample(self.n_docs)
        else: self.docs = docs
        self.n_docs = len(self.docs.index)
        self.docs.index.name = 'doc_id'

        # Parse docs into doc-word table
        stops = [w.strip() for w in open(self.stopwords_file, 'r').readlines()]
        docbow = []
        for idx in self.docs.index:
            doc_n = 0
            doc = self.docs.loc[idx].doc_content
            for word in re.split(r'\W+', doc):
                word = re.sub(r'\W+', '', word).lower().strip()
                if len(word) > 2 and word not in stops:
                    doc_n += 1
                    docbow.append((idx, doc_n, word))
        self.dw = pd.DataFrame(docbow, columns=['doc_id', 'word_pos', 'word_str'])
        
        # Add lengths to doc table
        self.docs['doc_length'] = self.dw.groupby('doc_id').word_str.count()
        
        # Create word table
        self.words = pd.DataFrame(dict(word_str=self.dw.word_str.sort_values().unique()))
        self.words.index.name = 'word_id'
        self.n_words = len(self.words.index)

        # Add word_id to doc-word table
        self.dw = self.dw.merge(self.words.reset_index(), on='word_str')
        self.dw.sort_values('doc_id', inplace=True)
        self.dw.reset_index(drop=True, inplace=True)
        
        # Remove duplicate words from docs
        self.dw = pd.DataFrame(self.dw.groupby(['doc_id', 'word_id']).size(), columns=['word_count'])

        # Add topics tables
        self.topics = pd.DataFrame(dict(topic_id=range(self.n_topics)))
        self.topics.set_index('topic_id', inplace=True)
        self.topics['topic_words'] = 'TBA'
        self.topic_ids = list(self.topics.index)
        
        # Initialize topic assignments to words
        self.dw['topic_id'] = self.dw.apply(lambda x: random.choice(self.topic_ids), 1)
        self.dw['topic_weight'] = 0     
        
        # Create doc-word co-occurrence matrix -- TURNS OUT NOT TO BE NECESSARY
        #self.dwm = self.dw.unstack().fillna(0).astype('int')
        #self.dwm.columns = self.dwm.columns.droplevel(0)
        #self.dwm[self.dwm > 0] = 1
    
        # Save data 
        self.save(self.docs, 'doc')
        self.save(self.topics, 'topic')
        self.save(self.words, 'word')
        self.save(self.dw, 'docword')
        #self.save(self.dwm, 'docword_matrix')

    def make_model(self):
        
        # Initialize topic assignments and weights
        self.dw['topic_id'] = self.dw.apply(lambda x: random.choice(self.topic_ids), 1)
        self.dw['topic_weight'] = 0.0

        # Iterate
        for iter in range(self.n_iters):

            # Show where you are            
            print('ITER', iter + 1)

            # Create matrices each loop to reflect the updated status of dw from previous loop
            self.decompose_dw()
            
            # Get best topic
            for idx in self.dw.index:
                doc_id = idx[0]
                word_id = idx[1]
                current_topic_id = int(self.dw.loc[idx].topic_id)               

                #self.decompose_dw()

                # Gibbs sampler
                self.dt.loc[doc_id, current_topic_id] -= 1
                self.tw.loc[current_topic_id, word_id] -= 1
                topic_weights = [0.0 for _ in self.topic_ids]
                for topic_id in self.topic_ids:
                    p_td = (self.dt.loc[doc_id, topic_id] + self.alpha) / (self.dt.loc[doc_id].sum() + self.n_topics * self.alpha)
                    p_wt = (self.tw.loc[topic_id, word_id] + self.beta) / (self.tw.loc[topic_id].sum() + self.n_words * self.beta)
                    topic_weights[topic_id] = p_td * p_wt
                self.dt.loc[doc_id, current_topic_id] += 1
                self.tw.loc[current_topic_id, word_id] += 1
                
                new_topic_id = self.sample_from(topic_weights)
                self.dw.loc[idx, 'topic_id'] = new_topic_id
                self.dw.loc[idx, 'topic_weight'] = topic_weights[new_topic_id]
                                
            # Periodically show results
            if iter % self.show_topics_interval == 0:
                self.add_topwords_to_topics_df()
                print(self.topics)
        
        self.save(self.dw, 'docword')
        
        # Get and save estimates φ’ and θ’ 
        self.make_theta()
        self.make_phi()

    def make_composite_matrix(self, df, cols, count_col):
        return df.groupby(cols)[count_col].count().unstack().fillna(0).astype('int')
    
    def decompose_dw(self):
        self.dt = self.make_composite_matrix(self.dw.reset_index(), ['doc_id', 'topic_id'], 'word_id')
        self.tw = self.make_composite_matrix(self.dw.reset_index(), ['topic_id', 'word_id'], 'doc_id')
        
    def make_theta(self):
        self.theta = self.dt.copy()
        for doc_id in self.dt.index:
            for topic_id in self.dt.columns:
                self.theta.loc[doc_id, topic_id] = (self.dt.loc[doc_id, topic_id] + self.alpha) / \
                (self.dt.loc[doc_id].sum() + self.n_topics * self.alpha)        
        self.theta = pd.DataFrame(self.theta.stack(), columns=['topic_weight'])
        self.save(self.theta, 'theta')
    
    def make_phi(self):
        self.phi = self.tw.copy()
        for topic_id in self.tw.index:
            for word_id in self.tw.columns:
                self.phi.loc[topic_id, word_id] = (self.tw.loc[topic_id, word_id] + self.beta) / \
                (self.tw.loc[topic_id].sum() + self.n_words + self.alpha)
        self.phi = pd.DataFrame(self.phi.stack(), columns=['word_freq'])
        self.save(self.phi, 'phi')
    
    def add_topwords_to_topics_df(self):
        topwords = self.tw.unstack().reset_index()
        topwords.columns = ['word_id','topic_id','weight']
        topwords = topwords.set_index('word_id').join(self.words).reset_index()
        for t in self.topic_ids:
            r = topwords[topwords.topic_id == t].sort_values('weight', ascending=False)[:self.n_topwords]
            self.topics.loc[t, 'topic_words'] = ', '.join(r.word_str.values)
        self.save(self.topics, 'topic')
        
    def sample_from(self, weights):
        """From Grus 2015"""
        rnd = sum(weights) * random.random()      
        for i, w in enumerate(weights):
            rnd -= w           
            if rnd <= 0:
                return i
                                                                      

if __name__ == '__main__':
    
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    tm = MyLDA()
    tm.make_corpus()
    tm.make_model()
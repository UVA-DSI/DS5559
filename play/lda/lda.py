# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
import random


class MyLDA:
    
    n_docs = 100
    sample = False
    corpus = 'corpus-small.txt'
    n_topics = 4
    n_iters = 100
    alpha = 0.5
    beta = 0.5
    n_topwords = 7
    show_topics_interval = 5
    
    stopwords_file = 'stopwords.txt'
    LETTERS = re.compile(r"[^a-z]")

    ###################
    ## DOCS HANDLING ##
    ###################
    
    def import_docs(self):
        docs = pd.read_csv(self.corpus)
        if self.sample: self.docs = docs.sample(self.n_docs)
        else: self.docs = docs
        self.n_docs = len(self.docs.index)
        self.docs.index.name = 'doc_id'
        
    def make_docwords_df(self):
        stops = [w.strip() for w in open(self.stopwords_file, 'r').readlines()]
        docbow = []
        for idx in self.docs.index:
            doc = self.docs.loc[idx].doc_content
            for word in re.split(r'\W+', doc):
                word = re.sub(r'\W+', '', word).lower().strip()
                if len(word) > 2 and word not in stops:
                    docbow.append((idx, word))
        self.dw = pd.DataFrame(docbow, columns=['doc_id', 'word_str'])
        self.docs['doc_length'] = self.dw.groupby('doc_id').word_str.count()

    def make_words_df(self):
        self.words = pd.DataFrame(dict(word_str=self.dw.word_str.sort_values().unique()))
        self.words.index.name = 'word_id'
        self.n_words = len(self.words.index)

    def add_word_ids_to_docwords_df(self):
        self.dw = self.dw.merge(self.words.reset_index(), on='word_str')
        self.dw.sort_values('doc_id', inplace=True)
        self.dw.reset_index(drop=True, inplace=True)

    #####################
    ## TOPICS HANDLING ##
    #####################
    
    def make_topics_df(self):
        self.topics = pd.DataFrame(dict(topic_id=range(self.n_topics)))
        self.topics.set_index('topic_id', inplace=True)
        self.topics['topic_words'] = 'TBA'
        self.topic_ids = list(self.topics.index)
                
    def initialize_topic_assignments(self):
        self.dw['topic_id'] = self.dw.apply(lambda x: random.choice(self.topic_ids), 1)
        self.dw['topic_weight'] = 0
        
    def make_composite_matrix(self, df, cols, count_col):
        comp = df.groupby(cols)[count_col].count().unstack().fillna(0)
        comp = comp.astype('int')
        return comp
    
    def make_composite_dfs(self):
        self.dt = self.make_composite_matrix(self.dw, ['doc_id', 'topic_id'], 'word_id')
        self.tw = self.make_composite_matrix(self.dw, ['topic_id', 'word_id'], 'doc_id')
        self.dtc = self.dt.sum(1)
        self.twc = self.tw.sum(1)
        
    def init_composite_dfs(self):
        "Uses Dirichlet priors; from Agustinus Kristiadi's Blog"
        self.dt = self.dt.apply(lambda x: np.random.dirichlet(self.alpha * np.ones(self.n_topics)), axis=1)
        self.tw = self.tw.apply(lambda x: np.random.dirichlet(self.beta * np.ones(self.n_words)), axis=1)
                
    def add_topwords_to_topics_df(self):
        topwords = self.tw.unstack().reset_index()
        topwords.columns = ['word_id','topic_id','weight']
        topwords = topwords.set_index('word_id').join(self.words).reset_index()
        for t in self.topic_ids:
            r = topwords[topwords.topic_id == t].sort_values('weight', ascending=False)[:self.n_topwords]
            self.topics.loc[t, 'topic_words'] = ', '.join(r.word_str.values)
        
    def sample_from(self, weights):
        """From Grus 2015"""
        rnd = sum(weights) * random.random()      
        for i, w in enumerate(weights):
            rnd -= w           
            if rnd <= 0:
                return i
                
    def get_topic_weight(self, doc_id, word_id, topic_id):
        """From Grus 2015"""
        p_td = (self.dt.loc[doc_id, topic_id] + self.alpha) / (self.dtc.loc[doc_id] + self.n_topics * self.alpha)
        p_wt = (self.tw.loc[topic_id, word_id] + self.beta)  / (self.twc.loc[topic_id] + self.n_words * self.beta)
        return p_td * p_wt
          
    def get_best_topic(self, doc_id, word_id, current_topic_id):
        """From Grus 2015"""
        self.dt.loc[doc_id, current_topic_id] -= 1
        self.tw.loc[current_topic_id, word_id] -= 1

        # New method       
        """From Andrew Brooks' Blog"""
        denom_a = self.dt.loc[doc_id].sum() + self.n_topics * self.alpha # sum(dt[d,]) + K * alpha
        denom_b = self.tw.sum(1) + self.n_words * self.beta # rowSums(wt) + length(vocab) * eta 
        topic_weights = list(((self.dt.loc[doc_id] + self.alpha) / denom_a) * ((self.tw.T.loc[word_id] + self.beta) / denom_b))

        # Old method
        # See http://brooksandrew.github.io/simpleblog/articles/latent-dirichlet-allocation-under-the-hood/
        # to replace next line with an apply statement
        #topic_weights = [self.get_topic_weight(doc_id, word_id, topic_id) for topic_id in self.topic_ids]
                
        self.dt.loc[doc_id, current_topic_id] += 1
        self.tw.loc[current_topic_id, word_id] += 1
        
        i = self.sample_from(topic_weights)
        #i = np.random.choice(self.topic_ids, 1, p=topic_weights)
        #i = np.argmax(topic_weights)
        
        tw = topic_weights[i]
        self.dw.loc[(doc_id, word_id), 'topic_weight'] = tw

        return i
    
    def generate_model(self):
        self.initialize_topic_assignments()
        for i in range(self.n_iters):                
            print('ITER', i+1)    

            self.make_composite_dfs()   
            self.dw['topic_id'] = self.dw.apply(lambda x: self.get_best_topic(x.doc_id, x.word_id, x.topic_id), 1)

            if i % self.show_topics_interval == 0:
                self.add_topwords_to_topics_df()
                print(self.topics)       
                
                self.theta = self.dt * self.alpha
                self.theta = self.theta.apply(lambda x: x / x.sum(), 1)
                self.phi = self.tw * self.beta
                self.phi = self.phi.apply(lambda x: x / x.sum(), 1)
                self.phi.columns = list(self.words.word_str)

        self.add_topwords_to_topics_df()
        print(self.topics)       
        print(self.theta)
                
        
    def sample_from_dw(self, doc_id, word_id, current_topic_id):
        """From Agustinus Kristiadi's Blog"""
        p_dw = np.exp(np.log(self.dt.loc[doc_id]) + np.log(self.tw.T.loc[word_id]))
        p_dw /= np.sum(p_dw)
        p_dw = list(p_dw)
        new_topic_id = np.random.multinomial(1, p_dw).argmax()
        return new_topic_id
    

    def generate_model2(self):
        
        # Initialize distributions
        self.initialize_topic_assignments()
        self.dw = self.dw.groupby(['doc_id', 'word_id']).size().unstack().fillna(0)
        self.dw = pd.DataFrame(self.dw.unstack(), columns=['topic_id'])
        self.dw = self.dw['topic_id'].stack()
        self.dt = self.make_composite_matrix(self.dw.reset_index(), ['doc_id', 'topic_id'], 'word_id')
        self.tw = self.make_composite_matrix(self.dw.reset_index(), ['topic_id', 'word_id'], 'doc_id')
        self.dtc = self.dt.sum(1)
        self.twc = self.tw.sum(1)
            
        for i in range(self.n_iters):                
            print('ITER', i+1)    
            self.dw['topic_id'] = self.dw.apply(lambda x: self.sample_from_dw(x.doc_id, x.word_id, x.topic_id), 1)
            self.make_composite_dfs()   
            
            self.dt = self.dt.apply(lambda x: np.random.dirichlet(self.alpha + x, 1))
            self.tw = self.tw.apply(lambda x: np.random.dirichlet(self.beta + x, 1))

            if i % self.show_topics_interval == 0:
                self.add_topwords_to_topics_df()
                print(self.topics)                       
 
        self.add_topwords_to_topics_df()
        print(self.topics)       
        print(self.theta)
                            
    def do_all_prep(self):
        self.import_docs()
        self.make_docwords_df()
        self.make_words_df()
        self.add_word_ids_to_docwords_df()
        self.make_topics_df()
    

if __name__ == '__main__':
    
    tm = MyLDA()
    tm.do_all_prep()
    #tm.generate_model2()    
    
    

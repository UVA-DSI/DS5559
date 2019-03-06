
# -*- coding: utf-8 -*-

# %% Set config
base_path = '/Users/rca2t/Dropbox/Courses/DSI/DS5559'
local_lib = base_path + '/lib'
db_file = base_path + '/experiments/naive-sentiment/data/winereviews.db'

# %% Set Hyperparameters
params = dict(
    qntile_B      = .1,
    qntile_A      = .9,
    n_sets        = 4,
    smooth_alpha  = 1,
    binary_counts = True
)

# %% Import libraries
import sqlite3
import pandas as pd
import sys; sys.path.append(local_lib)
import textman.textman as tx
import numpy as np
from numpy.random import randint

# %% Import raw DOC data
db = sqlite3.connect(db_file)
sql = "SELECT review_id AS doc_id, description AS doc_content, points FROM review"
docs = pd.read_sql(sql, db, index_col='doc_id').drop_duplicates()
db.close()

# %% Clip DOC table by quantile
bound_A = int(docs.points.quantile(params['qntile_A']))
bound_B = int(docs.points.quantile(params['qntile_B']))
docs = docs[(docs.points <= bound_B) | (docs.points >= bound_A)].copy()

# %% Convert DOC points feature to A and B labels
docs.loc[docs.points >= bound_A, 'doc_label'] = 'A'
docs.loc[docs.points <= bound_B, 'doc_label'] = 'B'

# %% ------------ TRAINING ------------

# %% Create training and test sets from DOC 
docs['set'] = randint(0,params['n_sets'], len(docs.index))
training = docs.query('set != 0').copy()
testing = docs.query('set == 0').copy()

# %% Compute number of docs in each set for each label
labels = training.doc_label.value_counts()\
    .to_frame().rename(columns={'doc_label':'docs_n'})
labels.index.name = 'label_id'

# %% Compute prior for labels based on DOC
N_docs = training.shape[0]
labels['prior'] = labels.docs_n / N_docs
#labels['logprior'] = np.log(labels.prior)

# %% Convert to Structured Text Model
tokens, vocab = tx.create_tokens_and_vocab(training, src_col='doc_content')

# %% Add sentiment column to tokens table
tokens = tokens.join(training.doc_label, on='doc_id')

# %% Create BOW as DTM
#bow = tokens.groupby(['doc_label', 'doc_id'])['term_id'].value_counts()\
#    .to_frame().rename(columns={'term_id':'n'})
#dtm = bow.n.unstack().fillna(0)
dtm = tokens.groupby(['doc_label', 'doc_id'])['term_id'].value_counts()\
    .to_frame().term_id.unstack().fillna(0)
    
# %% Use binary counts?
if params['binary_counts']:
    dtm[dtm > 0] = 1

# %% Experimental Likelihoods ////////////////////////
# Eliminates need for dtm above and Jurafsky block below
# tlc = term_label_counts
    
#if params['binary_counts']:
#    tlc = tokens.reset_index()\
#        .groupby(['doc_label','doc_id']).term_id.value_counts()\
#        .to_frame().rename(columns={'term_id':'n'})\
#        .reset_index()\
#        .groupby(['doc_label','term_id']).n.count()\
#        .to_frame().reorder_levels(['term_id','doc_label']).n.unstack().fillna(0)
#else:        
#    tlc = tokens.reset_index()\
#        .groupby('doc_label').term_id.value_counts()\
#        .to_frame().reorder_levels(['term_id','doc_label']).unstack().fillna(0)
#    tlc.columns = tlc.columns.droplevel()
#    
#LH = tlc + params['smooth_alpha']
#LH = tlc.div(tlc.sum() + len(tlc.index))

# %% Compute number of tokens per label
for label in labels.index:
    vocab[label+'_n'] = dtm.loc[label].sum() + params['smooth_alpha']

# %% Drop terms with no data
vocab = vocab.dropna()

# %% Create sentiment language models (likelihoods)
"""See Jurafsky for specific formula"""
N_terms = len(vocab.index) # This for the alpha_smoothing
for label in labels.index:
    col = label + '_n'
    N_terms_per_label = int(vocab[col].sum())
    vocab[label] = vocab[col].div(N_terms_per_label + N_terms)

# %% ------------ TESTING ------------

# %% Get TEST DOCs
tokens_test, vocab_test = tx.create_tokens_and_vocab(testing, idx=['doc_id', 'sent_id'], 
                                                     src_col='doc_content', drop=True)

# %% Add sentiment column to TEST DOCs
tokens_test = tokens_test.join(testing.doc_label, on='doc_id')

# %% Reconcile vocabuluaries (fix this)
tokens_test = tokens_test.rename(columns={'term_id':'bad_id'})
tokens_test['term'] = tokens_test.bad_id.map(vocab_test.term)
tokens_test['term_id'] = tokens_test.term.map(vocab.reset_index().set_index('term').term_id)    

# %% EXPERIMENT //////
#TEST = tokens_test.reset_index().set_index('doc_id')['term_id'].to_frame()
#TEST = TEST.join(LH, on='term_id')
#TEST = TEST.reset_index().set_index(['doc_id','term_id']).sort_index()
#TEST = TEST.stack().to_frame().rename(columns={0:'LH'})
#TEST.index.names = ['doc_id','term_id','doc_label']

# %% Map models to sentiments
for label in labels.index:
    tokens_test[label] = tokens_test.term_id.map(vocab[label])
    testing[label] = tokens_test.groupby('doc_id')\
        .apply(lambda x: x[label].product() * labels.loc[label, 'prior'])
        
# %% Predict
testing['predict'] = testing[['A','B']].idxmax(1)

# %% ------------ EVALUATION ------------

# %% Get diff
testing['diff'] = (testing.A - testing.B)**2 * np.sign(testing.A - testing.B)

# %% Predict
#testing.loc[testing['diff'] > 0, 'predict'] = 'A'
#testing.loc[testing['diff'] <= 0, 'predict'] = 'B'

# %% Results
testing['result'] = testing.doc_label == testing.predict

# %% Peformance
T, F = testing.result.value_counts()
grade = round(T/(T+F) * 100, 4)
CM = testing.reset_index().groupby(['doc_label','predict']).doc_id.count().unstack()

# %% Print results

def get_pad(keystr, maxpad=20):
    pad = (maxpad - len(keystr)) * '.'
    return pad

for key in params:
    pad = get_pad(key)
    print('{}:{}{}'.format(key, pad, params[key]))
print('Bounds:{}<={}, >={}'.format(get_pad('bounds'), bound_B, bound_A))
print('T:{}{}\nF:{}{}\nGrade:{}{}'.format(get_pad('x'),T,  get_pad('x'), F, get_pad('grade'), grade))
print()
print('Confusion Matrix:')
print(CM)

# %% Exploration
#vocab['diff'] = vocab.A - vocab.B

# %% 
#vocab[['term','diff']].sort_values('diff', ascending=False).head(10).plot(kind='barh', x='term')

# %% Also show that review length correlates with positivity
#docs['doc_len'] = docs.doc_content.str.len()
#docs.plot(kind='scatter', x='doc_len', y='points')
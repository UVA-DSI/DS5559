# -*- coding: utf-8 -*-

#%% Set config
base_path = '/Users/rca2t/Dropbox/Courses/DSI/DS5559'
local_lib = base_path + '/lib'
db_file = base_path + '/experiments/naive-sentiment/data/winereviews.db'

#%% Set Hyperparameters
params = dict(
    qntile_B=.1,
    qntile_A=.9,
    n_sets=4,
    smooth_alpha=1,
    binary_counts=True
)

#%% Import libraries
import pandas as pd
from numpy.random import randint
import sys; sys.path.append(local_lib)
import textman.textman as tx

#%% Import raw DOC data

try:
    docs = pd.read_csv('winereviews.csv', index_col='doc_id')
except FileNotFoundError:
    import sqlite3
    db = sqlite3.connect(db_file)
    sql = "SELECT review_id AS doc_id, description AS doc_content, points FROM review"
    docs = pd.read_sql(sql, db, index_col='doc_id').drop_duplicates()
    db.close()
    del([db, sql, sqlite3])
    docs.to_csv('winereviews.csv', index=True)

#%% Clip DOC table by quantile
bound_A = int(docs.points.quantile(params['qntile_A']))
bound_B = int(docs.points.quantile(params['qntile_B']))
docs = docs[(docs.points <= bound_B) | (docs.points >= bound_A)].copy()

#%% Convert DOC points feature to A and B labels
docs.loc[docs.points >= bound_A, 'doc_label'] = 'A'
docs.loc[docs.points <= bound_B, 'doc_label'] = 'B'
#docs.loc[docs.points <= 100, 'doc_label'] = 'A'
#docs.loc[docs.points < 92, 'doc_label'] = 'B'
#docs.loc[docs.points < 88, 'doc_label'] = 'C'
#docs.loc[docs.points < 85, 'doc_label'] = 'D'

#%% ------------ TRAINING ------------

#%% Split out training and test sets from DOC 
docs['set'] = randint(0,params['n_sets'], len(docs.index))
training = docs.query('set != 0').copy()
testing = docs.query('set == 0').copy()

#%% Get TOKEN and VOCAB from training corpus
tokens, _ = tx.create_tokens_and_vocab(training, src_col='doc_content')

#%% Add sentiment label to TOKEN
tokens = tokens.join(training.doc_label, on='doc_id')

#%% Create VOCAB from TOKEN
vocab = tokens.groupby('term_id').token_norm.value_counts()\
    .to_frame().rename(columns={'token_norm':'n'})
vocab = vocab.reset_index().rename(columns={'token_norm':'term_str'}).set_index('term_id')

#%% Adjust TOKENS
tokens = tokens.reset_index()[['doc_label', 'doc_id', 'term_id']]

#%% Compute PRIOR
priors = tokens.groupby('doc_label').doc_id.count()
priors = priors / priors.sum()

#%% Compute LIKELIHOOD
likelihoods = tokens.groupby(['doc_label'])\
    .term_id.value_counts()\
    .to_frame().rename(columns={'term_id':'n'})\
    .reset_index()
likelihoods = likelihoods.set_index(['term_id','doc_label']).n.unstack().fillna(0)
likelihoods = (likelihoods + params['smooth_alpha']).div(likelihoods.sum() + (len(vocab.index) * params['smooth_alpha']))

#%% --------------- TESTING --------------------

#%% Get test corpus
test = testing.doc_content.str.lower().str.split(r'\W+', expand=True)\
    .stack().reset_index().rename(columns={'level_1':'token_ord', 0:'term_str'})
test['term_id'] = test.term_str.map(vocab.reset_index().set_index('term_str').term_id)
test = test.dropna()

#%% Convert corpus to BOW
test_docs = test.groupby(['doc_id','term_id']).term_id.count()\
    .unstack().apply(lambda x: x.dropna().index.astype('int').tolist(), 1)\
    .to_frame().rename(columns={0:'bow'})
test_docs['doc_label'] = testing.doc_label
if params['binary_counts']:
    # set() forces BOW to consist of only one token for each term
    test_docs['bow'] = test_docs.bow.apply(lambda x: set(x))

#%% Compute POSTERIOR and make prediction
posteriors = test_docs.bow.apply(lambda x: likelihoods.loc[x].product() * priors)
test_docs['prediction'] = posteriors.T.idxmax()

#%% Evaluation 
test_docs['result'] = test_docs.doc_label == test_docs.prediction
T, F = test_docs.result.value_counts()
grade = round(T/(T+F) * 100, 4)
CM = test_docs.reset_index().groupby(['doc_label','prediction']).doc_id.count().unstack().fillna(0)

#%%
print("______________________")
print("      RESULTS")
print("----------------------")
print('Grade:', grade)
print("----------------------")
print("Confusion matrix:")
print(CM)
print("______________________")

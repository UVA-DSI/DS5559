import glob
import pandas as pd
import sqlite3
import re

pwd = '/Users/rca2t/Dropbox/Courses/DSI/DS5559/UVA_DSI_REPO/play/lda/corpora'

docs = []
for group in glob.glob('20news-18828/*'):
    doc_label =  group.split('/')[-1]
    for doc in glob.glob('{}/*'.format(group)):
        doc_id = doc.split('/')[-1]
        doc_lines = open(doc, 'r', encoding='utf-8', errors='ignore').readlines()
        doc_from = doc_lines[:1][0]
        doc_subject = doc_lines[1:2][0]
        doc_content = ''.join(doc_lines[2:])
        if 'From:' not in doc_from:
            continue
        doc_from = re.sub(r'^From:\s+', '', doc_from)
        doc_subject = re.sub(r'^Subject:\s+', '', doc_subject)
        docs.append((doc_id,doc_label,doc_from,doc_subject,doc_content))

df = pd.DataFrame(docs)
df.columns = ['doc_id','doc_label','doc_from','doc_subject','doc_content']
df = df.set_index('doc_id')
with sqlite3.connect(pwd + '/20news.db') as db:
    df.to_sql('doc', db, if_exists='replace')


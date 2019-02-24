# -*- coding: utf-8 -*-
  
import re
import pandas as pd
  
  
# Identify the source text (F0)
src_file = '2701-0.txt'
  
# Import the text as list of lines
lines = open(src_file, 'r', encoding='utf-8').readlines()
  
# Trim the cruft we identified
lines = lines[340:21964]
  
# Convert the lines into one big line, preserving line breaks
bigline = ''.join(lines)
  
# Split the bigline into paragraphs
paras = re.split(r'\n\n+', bigline)
  
# Break line into paragraphs
  
# Split by non-character
paras2 = []
for para in paras:
    tokens = re.split(r'\W+', para)
    paras2.append(tokens)
  
  
# Split by non-character but keep them
paras3 = []
for para in paras:
    tokens = re.split(r'(\W+)', para)
    paras3.append(tokens)
  
# Split by non-character using list comprehension
paras4 = [re.split(r'\W+', para) for para in paras]
  
# Try in Pandas
  
# Import paragraphs into a data frame
df = pd.DataFrame(paras, columns=['line'])
df.index.name = 'line_id'
df.line = df.line.str.strip()
  
# Tokenize using this one trick
df2 = df.line.str.split(r'\W+', expand=True)\
    .stack()\
    .to_frame()\
    .rename(columns={0: 'token'})
df2.index.names = ['line_id', 'token_id']
  
# Do a simple normalization
df2['norm'] = df2.token.str.lower()
  
# Get top N tokens
N = 30
df2['norm'].value_counts().head(N).sort_values().plot(kind='barh')
  
# Visualize dispersion plots of 'ahab' and 'whale'
(df2['norm'] == 'whale').astype('int').plot(figsize=(10, 1))
(df2['norm'] == 'ahab').astype('int').plot(figsize=(10, 1))

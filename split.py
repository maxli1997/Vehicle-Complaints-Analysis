import pandas as pd
import re

df = pd.read_csv('Manualcheckkeywordssep11.csv')
ndf = []

for i,r in df.iterrows():
    sentence = r['sentences'].strip('"')
    sentence = re.sub('[#*,"!()]', ' ', sentence)
    sentence = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',sentence)
    sentence = re.sub(re.escape('?'), ' ', sentence)
    sentence = re.sub('[\s]+', ' ', sentence)
    words = sentence.split(' ')
    for word in words:
        if word == '':
            continue
        word = word.strip('"*,.?!\'')
        ndf.append({'Sentence_id':i,'Word':word})

ndf = pd.DataFrame(ndf)

ndf.to_csv('split.csv',index=False)
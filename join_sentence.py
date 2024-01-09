import pandas as pd

df = pd.read_excel('ToyotaManualLabeled copy.xlsx')

ids = df.Sentence_id.unique()

s = []

for id in ids:
    temp = df[df['Sentence_id']==id]
    sentence = ' '.join([str(x) for x in (temp['Word'])])
    s.append(sentence)

ndf = pd.DataFrame(s)
ndf.columns=['Sentence']

ndf.to_csv('test.csv')



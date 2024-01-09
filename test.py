import pandas as pd

df = pd.read_csv("ToyotaManualLabeled2csv.csv",usecols=["Sentence_id","Word","Label"])

val = df.Label.unique()
print(val)


ndf = pd.DataFrame(columns=["sentence_id","word","label"])

for i,r in df.iterrows():
    if r['Label'] not in val:
        ndf = ndf.append({"sentence_id":r['Sentence_id'],"word":r['Word'],"label":'O'},ignore_index=True)
    else:
        ndf = ndf.append({"sentence_id":r['Sentence_id'],"word":r['Word'],"label":r['Label']},ignore_index=True)
#ndf.columns=["sentence_id","word","label"]

ndf.to_csv("test.csv",index=None)

import pandas as pd

df = pd.read_excel('count_Manualcheckkeywordssep11.xlsx',sheet_name='Sheet1')

ndf = []

for i,r in df.iterrows():
    r = list(set(list(r)))
    r.remove(0)
    ndf.append(r)

ndf = pd.DataFrame(ndf)
ndf.to_csv('removed.csv',index=False)
    
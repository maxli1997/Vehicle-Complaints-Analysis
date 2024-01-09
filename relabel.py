from numpy.core.numeric import NaN
import pandas as pd
import numpy


df1 = pd.read_excel('ToyotaManualLabeled.xlsx',sheet_name='Jackie')
df2 = pd.read_excel('ToyotaManualLabeled.xlsx',sheet_name='Meitang')

df3 = pd.read_excel('ToyotaManualLabeled.xlsx',sheet_name='Zifei')

id = df1['Sentence_id'].to_list()
wd = df1['Word'].to_list()
j = df1['Manual Label'].to_list()
m = df2['Manual Label'].to_list()
z = df3['Manual Label'].to_list()

t = []
d = []
i = 0
r = []

for a,b,c in zip(j,m,z):        
    if a!=b and a!=c and b!=c:
        d.append(id[i])
    i += 1
d = numpy.array(d)
d = d[~numpy.isnan(d)]
s = set(d)

i = 0
for a,b,c in zip(j,m,z):   
    if id[i] in s:
        t.append(a)    
    elif a!=b and a!=c and b!=c:
        continue
    elif a==b or a==c:
        t.append(a)
    else:
        t.append(b)
    i += 1
ndf = pd.DataFrame(columns=['sentence_id','word','label'])

i=0
u = ['B-CE','I-CE', 'B-BE', 'I-BE']
for a in t:
    if a in u:
        r.append(id[i])
    i+=1
r = numpy.array(r)
r = r[~numpy.isnan(r)]
s = set(r)

i = 0
tt = []
iid = []
wwd =[]
for a in t:   
    if id[i] in s:
        i+=1
        continue
    else:
        tt.append(a)
        iid.append(id[i])
        wwd.append(wd[i])
    i += 1

#print(iid)
ndf['label'] = tt
ndf['sentence_id'] = iid
ndf['word'] = wwd

ndf.to_csv('n_manual_label.csv',index=None)


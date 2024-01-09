import pandas as pd

df1 = pd.read_csv('COMPLAINTS_RECEIVED_2015-2019.txt',delimiter='\t',header=None)
df2 = pd.read_csv('COMPLAINTS_RECEIVED_2020-2021.txt',delimiter='\t',header=None)
df = pd.concat([df1,df2])
df = df.reset_index()
df.columns=[str(i) for i in range(0, 50)]
df = df.drop(columns=['0'])
search_df = pd.read_excel('ToyotaFastextInput.xlsx',sheet_name='Sheet1',usecols=['Complaints'])
complaints = search_df.Complaints.tolist()
ndf = df.loc[df['20'].isin([str(c).upper() for c in complaints])]
ndf.to_csv('FullInput.csv')
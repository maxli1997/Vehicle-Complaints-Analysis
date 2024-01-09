import pandas as pd

df = pd.read_csv('prediction-dise_fulltrain_manulwithtoutCE(avg).csv')
df.columns=['Sentence ID','Complaint']
new_df = []

for index, row in df.iterrows():
    for word in row['Complaint'].strip("[]").split(","):
        word = word.strip(" \{\}").split(": ")
        new_row = [row['Sentence ID'],word[0].strip("'"),word[1].strip("'")]
        new_df.append(new_row)
new_df = pd.DataFrame(new_df,columns=['Sentence_id','Word','Label'])

labels = ['B-E','I-E','B-C','I-C']
new_df = new_df[new_df['Label'].isin(labels)]

new_df.to_csv('converted2.csv',index=False)
#print(new_df)
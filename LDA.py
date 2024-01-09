import pandas as pd
import numpy as np

df = pd.read_csv("LDA_all_complaints.csv", usecols=['Dominant_Topic','Topic_Perc_Contrib','Keywords','Text'])

f = open("LDA_results.txt", "w")

for i in range(0,7):
    temp_df = df[df['Dominant_Topic']==i]
    sentences = "Topic " + str(i+1) + ": " + temp_df.iloc[0]['Keywords'] + '\n'
    f.write(sentences)
    total = temp_df['Text'].count()
    sentences = "The number of posts: " + str(total) + '\n'
    f.write(sentences)
    temp_df.sort_values(by='Topic_Perc_Contrib', ascending=False, inplace=True)
    for it,row in temp_df.iterrows():
        f.write(row['Text']+'\n')
    f.write("\n")

f.close()

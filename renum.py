import pandas as pd

df = pd.read_csv('disengage_label.csv')

row = df['sentence_id'].to_list()
n_row = []
r = row[0]
i = 1
for j in row:
    if j == r:
        n_row.append(i)
        continue
    else:
        i += 1
        r = j
        n_row.append(i)

df['sentence_id'] = n_row

df.to_csv('new_index_disengage.csv',index=None)

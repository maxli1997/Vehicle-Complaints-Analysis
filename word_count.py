import pandas as pd
import re

df = pd.read_csv('ToyotaFastextInput2.csv')

keywords = ['adaptive cruise control',
            'adaptive cruise',

            'emergency braking system', 
            'forward emergency braking', 
            'forward collision avoidance', 
            'forward collision alert', 
            'backward collision alert', 
            'automatic braking', 

            'lane keep', 
            'lane keep assist',

            'lane assist', 
            'lane assist warning', 

            'lane keeping',  
             
            'steering assist', 
            'ADAS', 
            'control transition', 
            'Forward Collision Warning', 
            'FCW', 
            'Automatic Emergency Braking', 
            'AEB', 
            'Lane Departure Warning',
            'LDW', 
            
            'LKA', 'Blind Spot Warning', 
            'BSW', 'automatic braking', 
            'automatic steering', 
            'automated driving', 
            'lane trace assist', 
            'super cruise', 
            'auto pilot', 
            'lane departure prevention', 
            'blind spot assist', 
            'rear cross traffic alert system', 
            'propilot Assist', 
            'autopilot', 
            'advanced driver assistance system', 
            'surround view camera', 
            'automatic high beams', 
            'rear cross traffic warning', 
            'driver monitoring', 
            'semi-automated parking assist', 
            'night vision detection', 
            'pedestrian detection', 
            'dynamic driving assistance', 
            'parking obstruction warning', 
            'automatic emergency steering', 
            'forward automatic emergency braking', 
            'reverse automatic emergency braking', 
            'fully-automated parking assistance', 
            'remote parking', 
            'pilot assist', 
            'pre-crash warning system', 
            'pre-crash braking system', 
            'intelligent braking system', 
            'forward collision mitigation system', 
            'smart cruise control', 
            'dynamic cruise control', 
            'lane centering assist', 
            'blind spot collision warning', 
            'cross traffic monitor']

new_columns = ['Label','Complaints']+keywords
new_df = []

for i,row in df.iterrows():
    new_row = []
    new_row.append(row['Label'])
    complaint = row['Complaints']
    new_row.append(complaint)
    for phrase in keywords:
        match= re.findall(phrase,complaint)
        length= len(match)
        new_row.append(length)
    new_df.append(new_row)

new_df = pd.DataFrame(new_df)
new_df.columns=new_columns
#ndf = pd.DataFrame(columns=new_columns)

'''for i,row in new_df.iterrows():
    row['cruise control'] = row['cruise control'] - row['adaptive cruise control']
    row['adaptive cruise'] = row['adaptive cruise'] - row['adaptive cruise control']
    row['cruise'] = row['cruise'] - row['adaptive cruise'] - row['cruise control'] - row['adaptive cruise control']
    row['lane keep'] = row['lane keep'] - row['lane keep assist']
    row['lane assist'] = row['lane assist'] - row['lane assist warning']
    ndf = ndf.append(row)'''

new_df.to_csv('count_results.csv',index=False)


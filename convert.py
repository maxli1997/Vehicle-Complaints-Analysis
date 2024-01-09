import pandas as pd

df = pd.read_excel('predictions1-2BeforeSplittingEngaDataWithoutCE.xlsx')
new_df = []

for index, row in df.iterrows():
    for word in row['Complaint'].strip("[]").split(","):
        word = word.strip(" \{\}").split(": ")
        new_row = [row['Sentence ID'],word[0].strip("'"),word[1].strip("'")]
        new_df.append(new_row)
new_df = pd.DataFrame(new_df,columns=['Sentence_id','Word','Label'])

#temp_df = new_df[new_df['Label']!='O']
#id = temp_df.Sentence_id.unique()

id = [2,20,30,44,68,72,72,79,86,86,92,112,113,119,129,134,138,141,141,144,144,154,174,181,182,185,201,214,214,248,250,252,267,280,280,282,283,288,293,303,304,350,381,387,395,399,418,424,452,470,474,477,477,495,501,513,513,514,518,522,529,530,531,532,552,559,581,587,590,594,630,638,640,647,648,655,658,658,679,705,728,776,783,791,798,799,799,817,826,840,882,883,890,891,897,919,929,934,950,965,970,987,1037,1037,1041,1051,1071,1091,1091,1098,1101,1125,1148,1178,1178,1200,1226,1239,1240,1240,1243,1246,1247,1254,1277,1284,1349,1350,1365,1365,1372,1372,1390,1397,1409,1444,1456,1462,1472,1472,1509,1513,1522,1524,1547,1548,1550,1563,1583,1592,1615,1623,1629,1629,1633,1654,1655,1655,1679,1681,1692,1692,1702,1713,1717,1720,1720,1741,1756,1774,1781,1783,1786,1786,1806,1807,1819,1819,1836,1865,1873,1901,1910,1915,1934,1937,1945,1960,1996,1998,2003,2003,2004,2004,2022,2045,2061,2078,2103,2108,2108,2109,2113,2119,2159,2172,2173,2181,2195,2195,2202,2206,2206,2206,2209,2218,2234,2235,2267,2276,2294,2298,2310,2322,2347,2374,2383,2389,2405,2405,2429,2432,2436,2450,2475,2494,2499,2500,2516,2525,2538,2551,2556,2556,2558,2574,2579,2579,2621,2636,2641,2642,2644,2645,2649,2656,2657,2657,2662,2668,2673,2682,2697,2699,2700,2705,2712,2716,2716,2726,2733,2736,2757,2776,2780,2791,2792,2797,2797,2799,2808,2829,2855,2859,2861,2879,2917,2936,2936,2959,2960,2975,2983,2985,2994,3017,3019,3026,3033,3049,3059,3060,3090,3090,3101,3104,3114,3127,3152,3161,3167,3174,3174,3184,3196,3199,3201,3214,3216,3221,3232,3235,3243,3245,3250,3251,3258,3268,3301,3341,3352,3400,3402,3402,3411,3411,3424,3434,3485,3492,3498,3532,3532,3548,3565,3565,3569,3579,3586,3595,3605,3623,3623,3630,3690,3693,3706,3713,3725,3728,3735,3746,3747,3756,3757,3766,3789,3790,3791,3798,3799,3800,3812,3813,3813,3818,3824,3825,3831,3836,3836,3841,3847,3852,3874,3874,3875,3876,3889,3894,3894,3897,3902,3911,3917,3929,3945,3950,3954,3955,3967,3968,4002,4015,4015,4032,4048
]

new_df = new_df[new_df['Sentence_id'].isin(id)]

new_df.to_csv('converted.csv',index=False)
print(new_df)
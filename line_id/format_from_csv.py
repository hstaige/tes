import os
import re
import numpy as np
import pandas as pd

match_csv = '/home/tim/research/apr24_data/theory/Bi_match.csv'
lev_dir = '/home/tim/research/apr24_data/theory/'
Z = 83

df = pd.read_csv(match_csv)
df = df[df['Energy'].notna()]
df = df.fillna(method='ffill')
df = df.reset_index(drop=True)
print(df)

header_z_re = re.compile(r'.*Z\s+=\s+([0-9]*)')
header_q_re = re.compile(r'NELE\s+=\s+([0-9]+)')
header_nlevs_re = re.compile(r'NLEV\s+=\s+([0-9]+)')
lev_dict = {}
levs = [f for f in os.listdir(lev_dir) if re.match(r'.*(\.lev)\Z',f)]
for f in levs:
    with open(lev_dir+f,'r+') as file:
        while True:
            line = file.readline()
            line_z_re = re.match(header_z_re,line)
            line_q_re = re.match(header_q_re,line)
            line_nlevs_re = re.match(header_nlevs_re,line)
            if line_z_re:
                z = line_z_re[1]
            elif line_q_re:
                q = int(z) - int(line_q_re[1]) + 1
            elif line_nlevs_re:
                nlevs = int(line_nlevs_re[1])
                break
        file.readline()
        lev_data = [file.readline()[85:] for _ in range(nlevs)]
    if int(z)==Z:
        lev_dict[q] = lev_data
        print(q)
print(lev_dict[54])
def label_term(row):
    return lev_dict[int(row['Charge'])][int(row['Upper_index']-1)]

df['Upper_term'] = df.apply(label_term, axis=1)

print(df)

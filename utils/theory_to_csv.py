import re
import pandas as pd

def format(line):
    line = re.sub(r'([0-9]+[a-z][0-9]+)\s([0-9]+[a-z][0-9]+)',r'\1.\2',line)
    line = re.sub(r'(0)\s([0-9]+[a-z][0-9]+)',r'\1.\2',line)
    line = re.sub(r'(0)\s([0-9]+[a-z]*)',r'\1.\2',line)
    line = re.sub(r'J=([0-9]+)',r'\1',line)
    line = re.sub(r'\s+',',',line)
    line = re.sub(r'([0-9]+)e',r'\1E',line)
    line = line.split(r',')
    return line

def theory_csv(input_dat, output_dest):
    with open(input_dat, 'r') as file:
        theory = file.readlines()

    theory = [re.sub(r':|-\s|\s\|','',line) for line in re.split(r'\|',''.join([i for sb in theory for i in sb]))]
    theory = [format(line) for line in theory]

    for i,line in enumerate(theory):
        if len(line)<12:
            theory[i][:0]=theory[i-1][:3]

    theory = [list(filter(None, line[:-1])) for line in theory]

    theory = pd.DataFrame(theory[:-1], columns=['Energy','Total_intesity','Charge','Lower_config','Lower_index','Lower_J','Upper_config','Upper_index','Upper_J','Intensity'])
    theory.to_csv(output_dest)


dir = '/home/tim/research/tes/theory'
theory_csv(f'{dir}/Nd.dat', f'{dir}/Nd.csv')
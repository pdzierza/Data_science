import json 

import numpy as np

import random

import argparse 

 
# Musimy wczytać parametry

def ParseArguments():
    parser = argparse.ArgumentParser(description="Motif generator")
    parser.add_argument('--params', default="params_set1.json", required=False, help='Plik z Parametrami  (default: %(default)s)')
    parser.add_argument('--output', default="generated_data.json", required=False, help='Plik z Parametrami  (default: %(default)s)')
    args = parser.parse_args()
    return args.params, args.output
    
    
param_file, output_file = ParseArguments()
 

with open(param_file, 'r') as inputfile:
    params = json.load(inputfile)
 
 
w=params['w']
k=params['k']
alpha=params['alpha']
Theta = np.asarray(params['Theta'])
ThetaB =np.asarray(params['ThetaB'])





wyniki=np.zeros((w,k)) #wymaga pozniej transpozycji
for i in range(w):
    if (random.random()>alpha):
        kolumnai=np.random.choice(4, k, p=np.transpose(Theta)[i])+1 #+1 skaluje do 1-4
        wyniki[i,:]=kolumnai
    else:
        kolumnai=np.random.choice(4, k, p=ThetaB)+1 #+1 skaluje do 1-4
        wyniki[i,:]=kolumnai

X=(wyniki.T).astype(int)






gen_data = {    
    "alpha" : alpha,
    "X" : X.tolist()
    }



with open(output_file, 'w') as outfile:
    json.dump(gen_data, outfile)
 

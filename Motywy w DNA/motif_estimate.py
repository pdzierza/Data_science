import json 

import numpy as np
 
import argparse 

 
# Musimy wczytać parametry

def ParseArguments():
    parser = argparse.ArgumentParser(description="Motif generator")
    parser.add_argument('--input', default="generated_data.json", required=False, help='Plik z danymi  (default: %(default)s)')
    parser.add_argument('--output', default="estimated_params.json", required=False, help='Tutaj zapiszemy wyestymowane parametry  (default: %(default)s)')
    parser.add_argument('--estimate-alpha', default="no", required=False, help='Czy estymowac alpha czy nie?  (default: %(default)s)')
    args = parser.parse_args()
    return args.input, args.output, args.estimate_alpha
    
    
input_file, output_file, estimate_alpha = ParseArguments()
 


with open(input_file, 'r') as inputfile:
    data = json.load(inputfile)
 
 
 
alpha=data['alpha']
X= np.asarray(data['X'])
X=X.astype(int)
k,w = X.shape




if (estimate_alpha == 'no'):
    TBnowe=np.random.uniform(0.1,0.3,4)
    TBnowe[3]=1-(TBnowe[0]+TBnowe[1]+TBnowe[2]) #poczatkowe ThetaB

    Tnowe=np.random.uniform(0.1,0.3,(4,w))
    Tnowe=Tnowe / Tnowe.sum(axis=0)             #poczatkowe Theta

    for b in range(1000):
        TBstare=TBnowe
        Tstare=Tnowe

        thetab=np.zeros(k)
        for i in range(k):
            iloczynb=np.zeros(w)
            for j in range(w):
                iloczynb[j]=TBstare[X[i,j]-1]# -1 bo wspolrzedne zaczynaja sie od 0
            thetab[i]=np.prod(iloczynb)
        theta=np.zeros(k)
        for i in range(k):
            iloczynb=np.zeros(w)
            for j in range(w):
                iloczynb[j]=Tstare[X[i,j]-1][j]# -1 bo wspolrzedne zaczynaja sie od 0
            theta[i]=np.prod(iloczynb)

        Qzero=(1-alpha)*thetab/((1-alpha)*thetab+alpha*theta)
        Qjeden=1-Qzero
        sumaQjeden=np.sum(Qjeden)

        sumajeden=np.zeros(k)
        sumadwa=np.zeros(k)
        sumatrzy=np.zeros(k)
        sumacztery=np.zeros(k)
    
        for i in range(k):
            sumajeden[i]=np.count_nonzero(X[i,:] == 1)
            sumadwa[i]=np.count_nonzero(X[i,:] == 2)
            sumatrzy[i]=np.count_nonzero(X[i,:] == 3)
            sumacztery[i]=np.count_nonzero(X[i,:] == 4)

        thetajedenb=np.sum(Qzero*sumajeden)/(w*np.sum(Qzero))
        thetadwab=np.sum(Qzero*sumadwa)/(w*np.sum(Qzero))
        thetatrzyb=np.sum(Qzero*sumatrzy)/(w*np.sum(Qzero))
        thetaczteryb=np.sum(Qzero*sumacztery)/(w*np.sum(Qzero))
        TBnowe=np.array([thetajedenb,thetadwab,thetatrzyb,thetaczteryb])

        for l in range(4):
            for j in range(w):
                Tnowe[l,j]=np.sum(Qjeden*(X[:,j] == l+1))/sumaQjeden

        #sprawdzanie zbieznosci
        bladTB=np.sqrt(np.sum((TBstare-TBnowe)**2)/4)
        bladT=np.sqrt(np.sum((Tstare-Tnowe)**2)/(4*w))
        if (bladTB<0.0001 and bladT<0.0001):
            break
    ThetaB=TBnowe
    Theta=Tnowe

elif (estimate_alpha == 'yes'):
    TBnowe=np.random.uniform(0.1,0.3,4)
    TBnowe[3]=1-(TBnowe[0]+TBnowe[1]+TBnowe[2])

    Tnowe=np.random.uniform(0.1,0.3,(4,w))
    Tnowe=Tnowe / Tnowe.sum(axis=0)

    alphanowe=np.random.uniform(0.1,0.5,1)

    for b in range(1000):
        TBstare=TBnowe
        Tstare=Tnowe
        alphastare=alphanowe

        thetab=np.zeros(k)
        for i in range(k):
            iloczynb=np.zeros(w)
            for j in range(w):
                iloczynb[j]=TBstare[X[i,j]-1]# -1 bo wspolrzedne zaczynaja sie od 0
            thetab[i]=np.prod(iloczynb)
        theta=np.zeros(k)
        for i in range(k):
            iloczynb=np.zeros(w)
            for j in range(w):
                iloczynb[j]=Tstare[X[i,j]-1][j]# -1 bo wspolrzedne zaczynaja sie od 0
            theta[i]=np.prod(iloczynb)

        Qzero=(1-alphastare)*thetab/((1-alphastare)*thetab+alphastare*theta)
        Qjeden=1-Qzero
        sumaQjeden=np.sum(Qjeden)

        sumajeden=np.zeros(k)
        sumadwa=np.zeros(k)
        sumatrzy=np.zeros(k)
        sumacztery=np.zeros(k)
        for i in range(k):
            sumajeden[i]=np.count_nonzero(X[i,:] == 1)
            sumadwa[i]=np.count_nonzero(X[i,:] == 2)
            sumatrzy[i]=np.count_nonzero(X[i,:] == 3)
            sumacztery[i]=np.count_nonzero(X[i,:] == 4)

        thetajedenb=np.sum(Qzero*sumajeden)/(w*np.sum(Qzero))
        thetadwab=np.sum(Qzero*sumadwa)/(w*np.sum(Qzero))
        thetatrzyb=np.sum(Qzero*sumatrzy)/(w*np.sum(Qzero))
        thetaczteryb=np.sum(Qzero*sumacztery)/(w*np.sum(Qzero))
        TBnowe=np.array([thetajedenb,thetadwab,thetatrzyb,thetaczteryb])

        for l in range(4):
            for j in range(w):
                Tnowe[l,j]=np.sum(Qjeden*(X[:,j] == l+1))/sumaQjeden

        alphanowe=np.sum(Qjeden)/np.sum(Qzero+Qjeden)

        #sprawdzanie zbieznosci
        bladTB=np.sqrt(np.sum((TBstare-TBnowe)**2)/4)
        bladT=np.sqrt(np.sum((Tstare-Tnowe)**2)/(4*w))
        bladalpha=np.abs(alphastare-alphanowe)
    
        if (bladTB<0.0001 and bladT<0.0001 and  bladalpha<0.0001):
            break
    ThetaB=TBnowe
    Theta=Tnowe
    alpha=alphanowe
else:
    print('Błędny argument w estimate-alpha.')


estimated_params = {
    "alpha" : alpha,            
    "Theta" : Theta.tolist(),   
    "ThetaB" : ThetaB.tolist()  
    }

with open(output_file, 'w') as outfile:
    json.dump(estimated_params, outfile)
    
    
    

import numpy as np
from sklearn.model_selection import train_test_split 
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
import random
import sys
import argparse
import os

parser = argparse.ArgumentParser(description="Projekt1 ")
parser.add_argument('--train', default="", required=True, help='zbior treningowy')
parser.add_argument('--test', default="", required=True, help='zbior testowy')
parser.add_argument('--alg', default="", required=True,type=str, help='nazwa metody')
parser.add_argument('--result', default="",type=str, required=True, help='plik wyjsciowy')


args = parser.parse_args()


treningowy=args.train
testowy=args.test
algorytm=args.alg
wyjsciowy=args.result



if algorytm=="NMF":
    data_trn=pd.read_csv(treningowy)
    data_tst=pd.read_csv(testowy)


    userId_trn=np.array(data_trn.userId)
    userId_trn=userId_trn.astype('int')-1 #przeskalowanie do pythonowego licznia od 0

    movieId_trn=np.array(data_trn.movieId)
    movieId_trn=movieId_trn.astype('int')-1 #przeskalowanie do pythonowego licznia od 0

    rating_trn=np.array(data_trn.rating)
    rating_trn=rating_trn.astype('float')

    userId_tst=np.array(data_tst.userId)
    userId_tst=userId_tst.astype('int')-1 #przeskalowanie do pythonowego licznia od 0

    movieId_tst=np.array(data_tst.movieId)
    movieId_tst=movieId_tst.astype('int')-1 #przeskalowanie do pythonowego licznia od 0

    rating_tst=np.array(data_tst.rating)
    rating_tst=rating_tst.astype('float')


    movieId_laczne=np.concatenate((movieId_trn,movieId_tst))

    Input1 = np.unique(movieId_laczne) 
    # using enumeate 
    temp = {y:x for x, y in enumerate(Input1)} 

    # List initialization 
    Input2 = movieId_laczne

    # Using list comprehension 
    Output = [temp.get(elem) for elem in Input2] 
    Output=np.asarray(Output,dtype=int)

    movieId_trn=Output[:len(movieId_trn)]
    movieId_tst=Output[len(movieId_trn):]

    n=len(np.unique(data_trn.userId))
    d=len(np.unique(Output))

    Z=np.zeros((n,d))

    Z[userId_trn,movieId_trn]=rating_trn



    sumaZ=Z.sum(axis=1, dtype='float')
    niezeroZ    = (Z != 0).sum(1)

    for i in range(n):
        Z[i,:]=sumaZ[i]
    Z=Z/niezeroZ[:,None]
    Z[userId_trn,movieId_trn]=rating_trn

    r=15
    model = NMF(n_components=r, init='random', random_state=0)
    W=model.fit_transform(Z)
    H=model.components_
    X_approximated = np.dot(W,H)  # macierz Z'
    #####tworzenie macierzy V (ktora tuaj znowu bedzie nazwana Z)

    Z=np.zeros((n,d))
    Z[userId_tst,movieId_tst]=rating_tst #macierz V

    wsp=tuple(zip(userId_tst,movieId_tst)) #wspolrzedne wystawionych ocen ze zbioru testowego
    suma=0
    for i in wsp:
        suma=suma+(X_approximated[i]-Z[i])**2

    suma=suma/len(userId_tst)
    suma=suma**0.5 #RMSE

    plk=open(wyjsciowy,"w+")
    plk.write(str(suma))
    plk.close()
    
    pre, ext = os.path.splitext(wyjsciowy)
    os.rename(wyjsciowy, pre + '.txt')
    
    
elif algorytm=="SVD1":
    data_trn=pd.read_csv(treningowy)
    data_tst=pd.read_csv(testowy)


    userId_trn=np.array(data_trn.userId)
    userId_trn=userId_trn.astype('int')-1 #przeskalowanie do pythonowego licznia od 0

    movieId_trn=np.array(data_trn.movieId)
    movieId_trn=movieId_trn.astype('int')-1 #przeskalowanie do pythonowego licznia od 0

    rating_trn=np.array(data_trn.rating)
    rating_trn=rating_trn.astype('float')

    userId_tst=np.array(data_tst.userId)
    userId_tst=userId_tst.astype('int')-1 #przeskalowanie do pythonowego licznia od 0

    movieId_tst=np.array(data_tst.movieId)
    movieId_tst=movieId_tst.astype('int')-1 #przeskalowanie do pythonowego licznia od 0

    rating_tst=np.array(data_tst.rating)
    rating_tst=rating_tst.astype('float')
    


    movieId_laczne=np.concatenate((movieId_trn,movieId_tst))

    Input1 = np.unique(movieId_laczne) 
    # using enumeate 
    temp = {y:x for x, y in enumerate(Input1)} 

    # List initialization 
    Input2 = movieId_laczne

    # Using list comprehension 
    Output = [temp.get(elem) for elem in Input2] 
    Output=np.asarray(Output,dtype=int)

    movieId_trn=Output[:len(movieId_trn)]
    movieId_tst=Output[len(movieId_trn):]

    n=len(np.unique(data_trn.userId))
    d=len(np.unique(Output))
    Z=np.zeros((n,d))

    Z[userId_trn,movieId_trn]=rating_trn



    sumaZ=Z.sum(axis=1, dtype='float')
    niezeroZ    = (Z != 0).sum(1)

    for i in range(n):
        Z[i,:]=sumaZ[i]
    Z=Z/niezeroZ[:,None]
    Z[userId_trn,movieId_trn]=rating_trn

    
    
    
    
    r=10
    svd=TruncatedSVD( n_components=r , random_state=42)
    svd.fit(Z)
    Sigma2=np.diag(svd.singular_values_)
    VT=svd.components_
    W=svd.transform(Z)/svd.singular_values_
    H= np.dot(Sigma2,VT)
    X_approximated = np.dot(W,H)
    #####tworzenie macierzy V (ktora tuaj znowu bedzie nazwana Z)

    Z=np.zeros((n,d))
    Z[userId_tst,movieId_tst]=rating_tst #macierz V

    wsp=tuple(zip(userId_tst,movieId_tst)) #wspolrzedne wystawionych ocen ze zbioru testowego
    suma=0
    for i in wsp:
        suma=suma+(X_approximated[i]-Z[i])**2

    suma=suma/len(userId_tst)
    suma=suma**0.5 #RMSE

    plk=open(wyjsciowy,"w+")
    plk.write(str(suma))
    plk.close()
    
    pre, ext = os.path.splitext(wyjsciowy)
    os.rename(wyjsciowy, pre + '.txt')
    
    
elif algorytm=="SVD2":
    data_trn=pd.read_csv(treningowy)
    data_tst=pd.read_csv(testowy)

    userId_trn=np.array(data_trn.userId)
    userId_trn=userId_trn.astype('int')-1 #przeskalowanie do pythonowego licznia od 0

    movieId_trn=np.array(data_trn.movieId)
    movieId_trn=movieId_trn.astype('int')-1 #przeskalowanie do pythonowego licznia od 0

    rating_trn=np.array(data_trn.rating)
    rating_trn=rating_trn.astype('float')

    userId_tst=np.array(data_tst.userId)
    userId_tst=userId_tst.astype('int')-1 #przeskalowanie do pythonowego licznia od 0

    movieId_tst=np.array(data_tst.movieId)
    movieId_tst=movieId_tst.astype('int')-1 #przeskalowanie do pythonowego licznia od 0

    rating_tst=np.array(data_tst.rating)
    rating_tst=rating_tst.astype('float')


    movieId_laczne=np.concatenate((movieId_trn,movieId_tst))

    Input1 = np.unique(movieId_laczne) 
    # using enumeate 
    temp = {y:x for x, y in enumerate(Input1)} 

    # List initialization 
    Input2 = movieId_laczne

    # Using list comprehension 
    Output = [temp.get(elem) for elem in Input2] 
    Output=np.asarray(Output,dtype=int)

    movieId_trn=Output[:len(movieId_trn)]
    movieId_tst=Output[len(movieId_trn):]

    n=len(np.unique(data_trn.userId))
    d=len(np.unique(Output))
    Z=np.zeros((n,d))

    Z[userId_trn,movieId_trn]=rating_trn



    sumaZ=Z.sum(axis=1, dtype='float')
    niezeroZ    = (Z != 0).sum(1)

    for i in range(n):
        Z[i,:]=sumaZ[i]
    Z=Z/niezeroZ[:,None]
    Z[userId_trn,movieId_trn]=rating_trn

    
    
    
    
    r=5

    svd=TruncatedSVD( n_components=r , random_state=42)
    svd.fit(Z)
    Sigma2=np.diag(svd.singular_values_)
    VT=svd.components_
    W=svd.transform(Z)/svd.singular_values_
    H= np.dot(Sigma2,VT)

    X_approximated = np.dot(W,H)
    X_approximated[userId_trn,movieId_trn]=rating_trn
    for m in range(14):
        svd=TruncatedSVD( n_components=r , random_state=42)
        svd.fit(X_approximated)
        Sigma2=np.diag(svd.singular_values_)
        VT=svd.components_
        W=svd.transform(X_approximated)/svd.singular_values_
        H= np.dot(Sigma2,VT)
        X_approximated = np.dot(W,H)
        X_approximated[userId_trn,movieId_trn]=rating_trn


      #####tworzenie macierzy V (ktora tuaj znowu bedzie nazwana Z)

    Z=np.zeros((n,d))
    Z[userId_tst,movieId_tst]=rating_tst #macierz V

    wsp=tuple(zip(userId_tst,movieId_tst)) #wspolrzedne wystawionych ocen ze zbioru testowego
    suma=0
    for i in wsp:
        suma=suma+(X_approximated[i]-Z[i])**2

    suma=suma/len(userId_tst)
    suma=suma**0.5 #RMSE

    plk=open(wyjsciowy,"w+")
    plk.write(str(suma))
    plk.close()
    
    pre, ext = os.path.splitext(wyjsciowy)
    os.rename(wyjsciowy, pre + '.txt')

    
elif algorytm=="SGD":
    data_trn=pd.read_csv(treningowy)
    data_tst=pd.read_csv(testowy)

    userId_trn=np.array(data_trn.userId)
    userId_trn=userId_trn.astype('int')-1 #przeskalowanie do pythonowego licznia od 0

    movieId_trn=np.array(data_trn.movieId)
    movieId_trn=movieId_trn.astype('int')-1 #przeskalowanie do pythonowego licznia od 0

    rating_trn=np.array(data_trn.rating)
    rating_trn=rating_trn.astype('float')

    userId_tst=np.array(data_tst.userId)
    userId_tst=userId_tst.astype('int')-1 #przeskalowanie do pythonowego licznia od 0

    movieId_tst=np.array(data_tst.movieId)
    movieId_tst=movieId_tst.astype('int')-1 #przeskalowanie do pythonowego licznia od 0

    rating_tst=np.array(data_tst.rating)
    rating_tst=rating_tst.astype('float')


    movieId_laczne=np.concatenate((movieId_trn,movieId_tst))

    Input1 = np.unique(movieId_laczne) 
    # using enumeate 
    temp = {y:x for x, y in enumerate(Input1)} 

    # List initialization 
    Input2 = movieId_laczne

    # Using list comprehension 
    Output = [temp.get(elem) for elem in Input2] 
    Output=np.asarray(Output,dtype=int)

    movieId_trn=Output[:len(movieId_trn)]
    movieId_tst=Output[len(movieId_trn):]

    n=len(np.unique(data_trn.userId))
    d=len(np.unique(Output))
    Z=np.zeros((n,d))


    Z[userId_trn,movieId_trn]=rating_trn

    r=15
    l=0.1  #lambda
    gamma=0.01

    W=np.random.rand(n,r)
    H=np.random.rand(r,d)
  
  

    for m in range(100):
        for u, v in tuple(zip(userId_trn,movieId_trn)):
            error = Z[u, v] - np.dot(W[u,:],H[:,v])
            W.T[:, u] = W.T[:, u] + gamma * (error * H[:, v] - l * W.T[:, u])
            H[:, v] = H[:, v] + gamma * (error * W.T[:, u] - l * H[:, v])
          
           
  
  

    X_approximated = np.dot(W,H)
    #####tworzenie macierzy V (ktora tuaj znowu bedzie nazwana Z)

    Z=np.zeros((n,d))


    Z[userId_tst,movieId_tst]=rating_tst
    wsp=tuple(zip(userId_tst,movieId_tst)) #wspolrzedne wystawionych ocen ze zbioru testowego
    suma=0
    for i in wsp:
        suma=suma+(X_approximated[i]-Z[i])**2  
    
    suma=suma/len(userId_tst)
    suma=suma**0.5 #RMSE

    plk=open(wyjsciowy,"w+")
    plk.write(str(suma))
    plk.close()
    
    pre, ext = os.path.splitext(wyjsciowy)
    os.rename(wyjsciowy, pre + '.txt')   
    
        
else:
    print("Nie ma takiego algorytmu.")
        


 

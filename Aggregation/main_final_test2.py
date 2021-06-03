   # -*- coding: utf-8 -*-
"""
Created on Wed May 22 00:18:30 2019

@author: yan
"""

from scipy.spatial.distance import pdist,squareform
import numpy as np
import time 
import pandas as pd
import numpy.matlib
from math import exp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score,precision_score,auc
from sklearn.metrics import accuracy_score,recall_score
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC,LinearSVC

start = time.time()

def Input():
    sample = pd.read_csv('Aggregation.csv',header=None)
    [N,L] = np.shape(sample)
    dim = L # Extract the num of dimensions
    # Extract the label of the data from the data frame
    label1 = sample.iloc[:,L-1]
    label = label1.values
    # Extract the samples from the data frame
    data = sample.iloc[:,0:dim-1]
    # Normalization Procedure
    NewData = Pre_Data(data)
    
#    FS = pd.read_csv('Selected_Feature.csv',header=None)
#
#    Ranking = FS.values
#    Ranking -= 1
    
#    ND = NewData[:,Ranking[0,:350]]
    ND = NewData
    return ND,label
#    return NewData,label 

def Pre_Data(data):
    [N,L] = np.shape(data)
    scaler = MinMaxScaler()
    scaler.fit(data)
    NewData = scaler.transform(data)
    return NewData

def ParamSpe(data):
    Buffersize = 788
    PreStd = []
    P_Summary = []
    PFS = []
    
    T = round(np.shape(data)[0]/Buffersize)
    
    return Buffersize,P_Summary,T,PFS,PreStd

def Distance_Cal(data):
    D = pdist(data)
    Dist = squareform(D)
    return Dist

def Fitness_Cal(sample,pop,stdData,gamma):
    Ns = np.shape(sample)[0]
    Np = np.shape(pop)[0]
    Newsample = np.concatenate([sample,pop])
    Dist = Distance_Cal(Newsample)
    fitness = []
    for i in range(Np):
        distArray = np.power(Dist[i+Ns,0:Ns],2)
        temp = np.power(np.exp(-distArray/stdData),gamma)
        fitness.append(np.sum(temp))
    return fitness

def fitness_update(P_Summary,Current,fitness,PreStd,gamma,stdData):
    [N,dim] = np.shape(Current)
    t_I = len(PreStd)
    NewFit = fitness
    if len(P_Summary)>0:
        PreFit = P_Summary[:,dim]
        PreP = P_Summary[:,0:dim]
        OldStd = PreStd[t_I-1]
        for i in range(N):
            fitin = 0
            for j in range(np.shape(PreP)[0]):
                if np.linalg.norm(Current[i][:]-PreP[j][:])<0.01:
                    fitin = PreFit[j]
                    break
                else:
                    d = np.linalg.norm(Current[i][:]-PreP[j][:])
                    fitin += (exp(-d**2/stdData)**gamma)*(PreFit[j]**(OldStd/stdData))
            NewFit[i] = fitness[i] + fitin
    return NewFit
    
def PopInitial(sample,PreMu,PreStd,Buffersize):
    [N,L] = np.shape(sample)
    pop_Size = round(1*N)
    # Compute the statistics of the current data chunk
    minLimit = np.min(sample,axis = 0)
    meanData = np.mean(sample,axis = 0)
    maxLimit = np.max(sample,axis =0)
    # Update the statistics of the data stream
    meanData = UpdateMean(PreMu,meanData,Buffersize)
    PreMu.append(meanData)
    # Compute the standard deviation of the current data chunk
    MD = np.matlib.repmat(meanData,N,1)
    tempSum = np.sum(np.sum((MD-sample)**2,axis=1))
    
    stdData = tempSum / N
#    stdData = 1
    # Update the standard deviation of the data stream
    stdData = StdUpdate(stdData,PreStd,Buffersize)
    # Randonmly Initialize the population indices from the data chunk
    pop_Index = np.arange(0,N)
    pop = sample[pop_Index,:]
    # Calculate the initial niche radius
    radius = numpy.linalg.norm((maxLimit-minLimit)) * 0.05 # 0.2
    
    return [stdData,pop_Index,pop,radius,PreMu,PreStd]

def UpdateMean(PreMy,meanData,BufferSize):
    # Num of the processed data chunk
    t_P = len(PreMu)
    # Update the mean of the data stream as new data chunk arrives
    if t_P==0:
        newMu = meanData
    else:
        oldMu = PreMu[t_P-1][:]
        newMu = (meanData + oldMu * t_P) / (t_P + 1)
    return newMu
    
def StdUpdate(Std,PreStd,BufferSize):
    # Num of the processed data chunk
    t_P = len(PreStd)
    # Update the variance of the data stream as new data chunk arrives
    if t_P==0:
        newStd = Std
    else:
        oldStd = PreStd[t_P-1]
        newStd = (Std + oldStd * t_P) / (t_P + 1)
    return newStd

##------------------------Parameter Estimation----------------------------#
#def CCA(sample,stdData,Dist):
#    m = 1
#    gamma = 5
#    ep = 0.975
#    N = np.shape(sample)[0]
#    while 1:
#        den1 = []
#        den2 = []
#        for i in range(N-1):
#            
#            Diff = np.power(Dist[i,:],2)
#            temp1 = np.power(np.exp(-Diff/stdData),gamma*m)
#            temp2 = np.power(np.exp(-Diff/stdData),gamma*(m+1))
#            den1.append(np.sum(temp1))
#            den2.append(np.sum(temp2))
#
#        My1 = np.mean(den1)
#        Stdy1 = np.std(den1)
#        My2 = np.mean(den2)
#        Stdy2 = np.std(den2)
#        
#        Th = min(Stdy1,Stdy2)
#        
#        if abs(My1-My2) < Th:
#            break
#        m = m + 1
#    return m*gamma

#------------------------Parameter Estimation----------------------------#
def CCA(sample,stdData,Dist):
    m = 1
    gamma = 5
    ep = 0.998 # 0.998
    N = np.shape(sample)[0]
    while 1:
        den1 = []
        den2 = []
        for i in range(N-1):
            
            Diff = np.power(Dist[i,:],2)
            temp1 = np.power(np.exp(-Diff/stdData),gamma*m)
            temp2 = np.power(np.exp(-Diff/stdData),gamma*(m+1))
            den1.append(np.sum(temp1))
            den2.append(np.sum(temp2))

        y = np.corrcoef(den1,den2)[0,1]
        
        if y > ep:
            break
        m = m + 1
    return m*gamma

#def DCCA(sample,stdData,P_Summary,gamma,dim):
#    P_Center = P_Summary[:,0:dim]
#    P_F = P_Summary[:,dim]
#    gam1 = gamma
#    N1 = np.shape(sample)[0]
#    N2 = np.shape(P_Center)[0]
#    ep = 0.975
#    N = N1 + N2
#    temp = np.concatenate([sample,P_Center],axis=0)
#    Dist = Distance_Cal(temp)
#    while 1:
#        gam2 = gam1 + 5
#        den1 = []
#        den2 = []
#        for i in range(N):
#            Diff = np.power(Dist[i,0:N1],2)
#            temp1 = np.power(np.exp(-Diff/stdData),gam1)
#            temp2 = np.power(np.exp(-Diff/stdData),gam2)
#            sum1 = np.sum(temp1)
#            sum2 = np.sum(temp2)
#            if i<N1:
#                T1 = 0
#                T2 = 0
#                for j in range(N2):
#                    T1 += P_F[j]**(gam1/gamma)
#                    T2 += P_F[j]**(gam2/gamma)
#                s1 = sum1 + T1
#                s2 = sum2 + T2
##                s1 = sum1**(gam1/gamma) + T1
##                s2 = sum2**(gam2/gamma) + T2
#            else:
##                s1 = sum1**(gam1/gamma) + P_F[i-N1]**(gam1/gamma)
##                s2 = sum2**(gam2/gamma) + P_F[i-N1]**(gam2/gamma)
#                s1 = sum1 + P_F[i-N1]**(gam1/gamma)
#                s2 = sum2 + P_F[i-N1]**(gam2/gamma)
#            den1.append(s1)
#            den2.append(s2)
#        My1 = np.mean(den1)
#        Stdy1 = np.std(den1)
#        My2 = np.mean(den2)
#        Stdy2 = np.std(den2)
#        
#        Th = min(Stdy1,Stdy2)
#        
#        if abs(My1-My2) < Th:
#            break
#        gam1 = gam2
#    return gam1

def DCCA(sample,stdData,P_Summary,gamma,dim):
    P_Center = P_Summary[:,0:dim]
    P_F = P_Summary[:,dim]
    gam1 = gamma
    N1 = np.shape(sample)[0]
    N2 = np.shape(P_Center)[0]
    ep = 0.998
    N = N1 + N2
    temp = np.concatenate([sample,P_Center],axis=0)
    Dist = Distance_Cal(temp)
    while 1:
        gam2 = gam1 + 5
        den1 = []
        den2 = []
        for i in range(N):
            Diff = np.power(Dist[i,0:N1],2)
            temp1 = np.power(np.exp(-Diff/stdData),gam1)
            temp2 = np.power(np.exp(-Diff/stdData),gam2)
            sum1 = np.sum(temp1)
            sum2 = np.sum(temp2)
            if i<N1:
                T1 = 0
                T2 = 0
                for j in range(N2):
                    T1 += P_F[j]**(gam1/gamma)
                    T2 += P_F[j]**(gam2/gamma)
                s1 = sum1 + T1
                s2 = sum2 + T2
#                s1 = sum1**(gam1/gamma) + T1
#                s2 = sum2**(gam2/gamma) + T2
            else:
#                s1 = sum1**(gam1/gamma) + P_F[i-N1]**(gam1/gamma)
#                s2 = sum2**(gam2/gamma) + P_F[i-N1]**(gam2/gamma)
                s1 = sum1 + P_F[i-N1]**(gam1/gamma)
                s2 = sum2 + P_F[i-N1]**(gam2/gamma)
            den1.append(s1)
            den2.append(s2)
        y = np.corrcoef(den1,den2)[0,1]
        if y > ep:
            break
        gam1 = gam2
    return gam1

def TPC_Search(Dist,Pop_Index,Pop,radius,fitness):
    # Extract the size of the population
    [N,dim] = np.shape(Pop)
    P = [] # Initialize the Peak Vector
    P_fitness = []
    i = 0
    marked = []
    co = []
    OriginalIndice = Pop_Index
    OriFit = fitness
    TPC_Indice = OriginalIndice
    PeakIndices = []
    while 1:
        #-------------Search for the local maximum-----------------#
        SortIndice = np.argsort(fitness)
        NewIndice = SortIndice[::-1]
        
        
        Pop = Pop[NewIndice,:]
        fitness = fitness[NewIndice]
        OriginalIndice = OriginalIndice[NewIndice]
        
#        PeakIndices.append(NewIndice[0])
        
        P.append(Pop[0,:])
        P_fitness.append(fitness[0])
        
        PeakIndices.append(np.where(OriFit==fitness[0])[0][0])
            
#        P_fitness.append(fitness[0])
        P_Indice = OriginalIndice[0]
        
        Ind = AssigntoPeaks(Pop,Pop_Index,P,P_Indice,marked,radius,Dist)
        
        marked.append(Ind)
        marked.append(NewIndice[0])
        
        
        
        if not Ind:
            Ind = [NewIndice[0]]
            
        TPC_Indice[Ind] = PeakIndices[i]
#        TPC_Indice[Ind] = P_Indice
        
        co.append(len(Ind))
        TempFit = fitness
        sum1 = 0
        for j in range(len(Ind)):
            sum1 += fitness[np.where(OriginalIndice==Ind[j])[0]]
        for th in range(len(Ind)):
            TempFit[np.where(OriginalIndice==Ind[th])[0]] = fitness[np.where(OriginalIndice==Ind[th])[0]]/(1+sum1)
        fitness = TempFit
        i = i + 1
        if np.sum(co)>=N:
#            PeakIndices = np.unique(PeakIndices)
#            P = sample[PeakIndices][:]
            P = np.asarray(P)
#            P_fitness = fitness[PeakIndices]
            P_fitness = np.asarray(P_fitness)
            TPC_Indice = Close_Clusters(pop,PeakIndices,Dist)
            break
        
           
    return P,P_fitness,TPC_Indice,PeakIndices

def Boundary_Instance(Current,Neighbor,Dist,TPC_Indice,sample):
    temp_cluster1 = np.where(TPC_Indice==Current)[0]
    temp_cluster2 = np.where(TPC_Indice==Neighbor)[0]
    temp = np.concatenate([temp_cluster1,temp_cluster2])
    
    Dc = Dist[Current][Neighbor]
    Dd = []
    
    for i in range(len(temp)):
        D1 = Dist[Current][temp[i]]
        D2 = Dist[Neighbor][temp[i]]
        Dd.append(abs(D1+D2-Dc))
    if len(Dd) == 0:
        BD = sample[Current][:]
    else:
        CI = np.argmin(Dd)
        BD = sample[temp[CI]][:]
    
    return BD
        
def MergeInChunk(P,P_fitness,sample,gamma,stdData,Dist,TPC_Indice,PeakIndices):
    """Perform the Merge of TPCs witnin each data chunk
    """
    # Num of TPCs
    [Nc,dim] = np.shape(P)
    NewP = []
    NewP_fitness = []
    marked= []
    unmarked= []
    Com = []
    
    # Num of TPCs
    Nc = np.shape(P)[0]
    for i in range(Nc):
        MinDist = np.inf
        MinIndice = 100000
        if i not in marked:
            for j in range(Nc):
                if j!=i and j not in marked:
                    d = np.linalg.norm(P[j,:]-P[i,:])
                    if d < MinDist:
                        MinDist = d
                        MinIndice = j
            if MinIndice <= Nc:
                MinIndice = int(MinIndice)
                Merge = True
#                Neighbor = PeakIndices[MinIndice]
#                Current = PeakIndices[i]
                Neighbor = P[MinIndice][:]
                X = (Neighbor + P[i,:])/2
                
#                X = Boundary_Instance(Current,Neighbor,Dist,TPC_Indice,sample)
                    
                X = np.reshape(X,(1,np.shape(P)[1]))
                    
                fitX = Fitness_Cal(sample,X,stdData,gamma)
                fitP = P_fitness[i]
                fitN = P_fitness[MinIndice]
                if fitX < 1*min(fitN,fitP):
                    Merge = False
                if Merge:
                    Com.append([i,MinIndice])
                    marked.append(MinIndice)
                    marked.append(i)
                else:
                    unmarked.append(i)
    Com = np.asarray(Com)
    # Number of Possible Merges:
    Nm = np.shape(Com)[0]
    for k in range(Nm):
        if P_fitness[Com[k,0]] >= P_fitness[Com[k,1]]:
            NewP.append(P[Com[k,0],:])
            NewP_fitness.append(P_fitness[Com[k,0]])
        else:
            NewP.append(P[Com[k,1],:])
            NewP_fitness.append(P_fitness[Com[k,1]])
    # Add Unmerged TPCs to the NewP
    for n in range(Nc):
        if n not in Com:
            NewP.append(P[n,:])
            NewP_fitness.append(P_fitness[n])
    NewP = np.asarray(NewP)
    NewP_fitness = np.asarray(NewP_fitness)
    return NewP,NewP_fitness

#def MergeInChunk(P,P_fitness,sample,gamma,stdData):
#    """Perform the Merge of TPCs witnin each data chunk
#    """
#    # Num of TPCs
#    [Nc,dim] = np.shape(P)
#    NewP = []
#    NewP_fitness = []
#    marked= []
#    unmarked= []
#    Com = []
#    
#    # Num of TPCs
#    Nc = np.shape(P)[0]
#    for i in range(Nc):
#        MinDist = np.inf
#        MinIndice = 100000
#        if i not in marked:
#            for j in range(Nc):
#                if j!=i and j not in marked:
#                    d = np.linalg.norm(P[j,:]-P[i,:])
#                    if d < MinDist:
#                        MinDist = d
#                        MinIndice = j
#            if MinIndice <= Nc:
#                MinIndice = int(MinIndice)
#                Merge = True
#                Neighbor = P[MinIndice][:]
#                X = (Neighbor + P[i,:])/2
#                    
#                X = np.reshape(X,(1,np.shape(P)[1]))
#                    
#                fitX = Fitness_Cal(sample,X,stdData,gamma)
#                fitP = P_fitness[i]
#                fitN = P_fitness[MinIndice]
#                if fitX < 1*min(fitN,fitP):
#                    Merge = False
#                if Merge:
#                    Com.append([i,MinIndice])
#                    marked.append(MinIndice)
#                    marked.append(i)
#                else:
#                    unmarked.append(i)
#    Com = np.asarray(Com)
#    # Number of Possible Merges:
#    Nm = np.shape(Com)[0]
#    for k in range(Nm):
#        if P_fitness[Com[k,0]] >= P_fitness[Com[k,1]]:
#            NewP.append(P[Com[k,0],:])
#            NewP_fitness.append(P_fitness[Com[k,0]])
#        else:
#            NewP.append(P[Com[k,1],:])
#            NewP_fitness.append(P_fitness[Com[k,1]])
#    # Add Unmerged TPCs to the NewP
#    for n in range(Nc):
#        if n not in Com:
#            NewP.append(P[n,:])
#            NewP_fitness.append(P_fitness[n])
#    NewP = np.asarray(NewP)
#    NewP_fitness = np.asarray(NewP_fitness)
#    return NewP,NewP_fitness

def MergeOnline(P,P_fitness,P_summary,PreStd,sample,gamma,stdData):
    """Perform the Merge of Clusters Between Historical and New Clusters
    """
    # Num of TPCs
    [Nc,dim] = np.shape(P)
    NewP = []
    NewP_fitness = []
    marked= []
    unmarked= []
    Com = []
    
    
    for i in range(Nc):
        MinDist = np.inf
        MinIndice = 100000
        if i not in marked:
            for j in range(Nc):
                if j!=i and j not in marked:
                    d = np.linalg.norm(P[j,:]-P[i,:])
                    if d < MinDist:
                        MinDist = d
                        MinIndice = j
            if MinIndice < Nc:

#                MinIndice = int(MinIndice)
                Merge = True
                Neighbor = P[MinIndice][:]
                X = (Neighbor + P[i][:])/2
                X = np.reshape(X,(1,np.shape(P)[1]))
                RfitX = Fitness_Cal(sample,X,stdData,gamma)
                fitX = fitness_update(P_Summary,X,RfitX,PreStd,gamma,stdData)
                fitP = P_fitness[i]
                fitN = P_fitness[MinIndice]
                if fitX < 1*min(fitN,fitP):
                    Merge = False
                if Merge:
                    Com.append([i,MinIndice])
                    marked.append(MinIndice)
                    marked.append(i)
                else:
                    unmarked.append(i)
    Com = np.asarray(Com)        
    # Number of Possible Merges:
    Nm = np.shape(Com)[0]
    for k in range(Nm):
        if P_fitness[Com[k,0]] >= P_fitness[Com[k,1]]:
            NewP.append(P[Com[k,0]][:])
            NewP_fitness.append(P_fitness[Com[k,0]])
        else:
            NewP.append(P[Com[k,1]][:])
            NewP_fitness.append(P_fitness[Com[k,1]])
    # Add Unmerged TPCs to the NewP
    for n in range(Nc):
        if n not in Com:
            NewP.append(P[n][:])
            NewP_fitness.append(P_fitness[n])
    NewP = np.asarray(NewP)
    NewP_fitness = np.asarray(NewP_fitness)
    return NewP,NewP_fitness

def CE_InChunk(sample,P,P_fitness,stdData,gamma,Dist,TPC_Indice,PeakIndices):
    while 1:
        HistP = P
#        HistPF = P_fitness
        P,P_fitness = MergeInChunk(P,P_fitness,sample,gamma,stdData,Dist,TPC_Indice,PeakIndices)
        if np.shape(P)[0] == np.shape(HistP)[0]:
            break
    return P,P_fitness

def CE_Online(sample,P_Summary,P,P_fitness,stdData,gamma,PreStd):
    dim = np.shape(P)[1]
    
    # Concatenate the historical and new clusters together
    PC = np.concatenate([P_Summary[:,0:dim],P])
    RPF = Fitness_Cal(sample,PC,stdData,gamma)
    PF = fitness_update(P_Summary,PC,RPF,PreStd,gamma,stdData)
    
    while 1:
        HistPC = PC
#        HistPF = PF
        PC,PF = MergeOnline(PC,PF,P_Summary,PreStd,sample,gamma,stdData)
        RPF = Fitness_Cal(sample,PC,stdData,gamma)
        PF = fitness_update(P_Summary,PC,RPF,PreStd,gamma,stdData)
        if np.shape(PC)[0] == np.shape(HistPC)[0]:
            break
    return PC,PF

def ClusterValidation(sample,P):
    while 1:
        NewP = []
        PreP = P
        [R_d,RIndice] = Cluster_Assign(sample,P)
        
        for i in range(np.shape(P)[0]):
            Temp = np.where(RIndice==i)
            Temp = np.asarray(Temp)
            if np.shape(Temp)[1]>2:
                NewP.append(P[i][:])
        P = NewP
        if np.shape(P)[0] == np.shape(PreP)[0]:
            break
    return np.asarray(P)

def ClusterSummary(P,PF,P_Summary,sample):
    dim = np.shape(sample)[1]
    Rp = AverageDist(P,P_Summary,sample,dim)
    P = np.asarray(P)
    PF = [PF]

    PF = np.asarray(PF)
    Rp = np.reshape(Rp,(np.shape(P)[0],1))
    PCluster = np.concatenate([P,PF.T],axis=1)
    PCluster = np.concatenate([PCluster,Rp],axis=1)
    
    
    P_Summary = PCluster
    
    
    return P_Summary

def StoreInf(PF,PFS,PreStd,stdData):
    PreStd.append(stdData)
    PFS.append(PF)
    return PreStd,PFS
    

#--------------------Cluster Radius Computation and Update--------------------#
def AverageDist(P, P_Summary, sample, dim):
    P = P
    # Obtain the assignment of clusters
    [distance,indices] = Cluster_Assign(sample,P)
    rad1 = []
    # if the summary of clusters is not empty
    if len(P_Summary)>0:
        
        PreP = P_Summary[:,0:dim] # Hstorical Cluster Center vector
        PreR = P_Summary[:,dim+1]
        for i in range(np.shape(P)[0]):
            if np.shape(np.where(indices==i))[1] >1:
                SumD1 = 0
                Count1 = 0
                for j in range(np.shape(sample)[0]):
                    if indices[j] == i:
                        SumD1 += distance[j]
                        Count1 += 1
                rad1.append(SumD1 / Count1)
            else:
                C_d = []
                for k in range(np.shape(PreP)[0]):
                    C_d.append(np.linalg.norm(P[i][:] - PreP[k][:]))
                CI = np.argmin(C_d)
                rad1.append(PreR[CI])
    elif not P_Summary:
        for i in range(np.shape(P)[0]):
            SumD1 = 0
            Count1 = 0
            for j in range(np.shape(sample)[0]):
                if indices[j] == i:
                    SumD1 += distance[j]
                    Count1 += 1
            rad1.append(SumD1/Count1)
    return np.asarray(rad1)

def AssigntoPeaks(pop,pop_index,P,P_I,marked,radius,Dist):
    temp = []
    [N,L] = np.shape(pop)
    for i in range(N):
        distance = Dist[i,P_I]
        if not np.any(marked==pop_index[i]):
            if distance <= radius:
                temp.append(pop_index[i])
    indices = temp
    return indices

def Close_Clusters(pop,PeakIndices,Dist):
    P = pop[PeakIndices][:]
    C_Indices = np.arange(0,np.shape(pop)[0])
    for i in range(np.shape(pop)[0]):
        temp_dist = Dist[i][PeakIndices]
        C_Indices[i] = PeakIndices[np.argmin(temp_dist)]
    return C_Indices
        

def Cluster_Assign(sample,P):
    # Number of samples
    N = np.shape(sample)[0]
    # Number of Clusters at t
    Np = np.shape(P)[0]
    MinDist = []
    MinIndice = []
    for i in range(N):
        d = []
        for j in range(Np):
            d.append(np.linalg.norm(sample[i][:]-P[j][:]))
        if len(d)<=1:
            tempD = d
            tempI = 0
        else:
            tempD = np.min(d)
            tempI = np.argmin(d)
            
        MinDist.append(tempD)
        MinIndice.append(tempI) 
    MinDist = np.asarray(MinDist)
    MinIndice = np.asarray(MinIndice)
    return MinDist,MinIndice
#---------------------New Function/Unfinished----------------------#
def onlineAL(P_summary,sample,sample_labels,t):
    [Ns,D] = np.shape(sample)
    cluster_centers = P_summary[:][:D]
    center_labels = P_summary[:][D+2]
    
    n = 0.1 * N
    
    [MinDist,ClusterIndice] = Cluster_Assign(sample,cluster_centers)
    
    FetchIndex = []
    UnlabeledIndex = []
    
    for i in range(np.shape(P)[0]):
        tempcluster = np.where(ClusterIndice==(i))
        d = []
        for j in range(len(tempcluster[0])):
            d.append(np.linalg.norm(AccSample[tempcluster[0][j],:]-P[i,:]))    
        sortIndex = np.argsort(d)
        
        fetchSize = num_S * len(d)/np.shape(AccSample)[0]
        
        fet1 = tempcluster[0][sortIndex[:round(fetchSize/2)]]
        fet1 = fet1.astype(int)
        fet2 = tempcluster[0][sortIndex[-round(fetchSize/2):]]
        fet2 = fet2.astype(int)
        FetchIndex = np.append(FetchIndex,fet1)
        FetchIndex = np.append(FetchIndex,fet2)  
        
        UnlabeledIndex = np.append(UnlabeledIndex,tempcluster[0][sortIndex[round(fetchSize/2):-round(fetchSize/2)]])

    FetchIndex = FetchIndex.astype(int)
    UnlabeledIndex = UnlabeledIndex.astype(int)

    sample_Fetch = np.append(OriData[FetchIndex][:],cluster_centers)
    sample_Unlabeled = np.append(OriData[UnlabeledIndex][:],center_labels)
    
    clf2 = MLPClassifier()
    clf2 = clf2.fit(sample_Fetch,label[FetchIndex])
    
    Acc2 = clf2.score(sample_Unlabeled,label[UnlabeledIndex])
    
    return Acc2,clf2

def Post_Sampling(fetchX,fetchY,p):
    dist = pdist(fetchX)
    D =  squareform(dist)
    reprsentative_power = []
    
    [Ns,dim] = np.shape(fetchX)
    
    for i in range(Ns):
        temp_dist =  D[i,:]
        sort_idx = np.argsort(temp_dist)
        temp_den = 1/( 1 + np.sum(temp_dist[sort_idx[:p]]))
        inconsist_idx = np.where(fetchY[sort_idx[:p]]!=fetchY[i])
        temp_count = 1 + np.shape(inconsist_idx)[1]
        
        reprsentative_power.append(temp_count * temp_den)
    
    return reprsentative_power    

#---------------------------Main Function-------------------------#
if __name__ == '__main__':
    [data,label] = Input()
    
    #data,test_x,label,test_y = train_test_split(data1, label1, test_size=0.25, random_state=42)
    
    [BufferSize,P_Summary,T,PFS,PreStd] = ParamSpe(data)
    
    T = int(T)
    gammaHist = []
    PFS = []
    PreMu = []
    for t in range(T):
        
        if t < T-1:
            sample = data[t*BufferSize:(t+1)*BufferSize,:]
        else:
            sample = data[t*BufferSize:np.shape(data)[0]]
        if t==0:
            AccSample = sample
        else:
            AccSample = np.concatenate([AccSample,sample])
        
        dim = np.shape(sample)[1]
        [stdData,pop_index,pop,radius,PreMu,PreStd] = PopInitial(sample,PreMu,PreStd,BufferSize)
        # Initialize the fitness vector
        fitness = np.zeros((len(pop_index),1))
        # Initialize the indices vector
        indices = np.zeros((len(pop_index),1))
        
        Dist = Distance_Cal(sample)
        if PreStd:
            if PreStd[len(PreStd)-1] > stdData:
                
                P = P_Summary[:,0:dim]
                
                localFit = Fitness_Cal(sample,P,stdData,gamma)
                
                PF = fitness_update(P_Summary,P,localFit,PreStd,gamma,stdData)
                P_Summary = ClusterSummary(P,PF,P_Summary,sample)
                PFS.append(PF)
                PreStd.append(stdData)
                clustercenter = P
                [Assign,clusterindex] = Cluster_Assign(AccSample,P)
                continue
#            gam = gamma
#            gamma = DCCA(sample,stdData,P_Summary,gam,dim)

        else:
            gamma = CCA(sample,stdData,Dist)
            
        gammaHist.append(gamma)
#        gamma = 20
        
        fitness = Fitness_Cal(sample,pop,stdData,gamma)
        fitness = np.array(fitness)
        
        P, P_fitness, TPC_Indice, PeakIndices = TPC_Search(Dist,pop_index,pop,radius,fitness)
        
        
        P, P_fitness = CE_InChunk(sample,P,P_fitness,stdData,gamma,Dist,TPC_Indice,PeakIndices)
        
        
#        P = ClusterValidation(sample,P)
        
        P_fitness = Fitness_Cal(sample,P,stdData,gamma)
        P_fitness = fitness_update(P_Summary,P,P_fitness,PreStd,gamma,stdData)
        
#        print('Error 2')
        
        print(t)
        
#        center_label = center_labelfetch(sample,P,sample_label)
#        P_Summary = ClusterSummary(P,PF,P_Summary,sample,sample_label)
        
        if t == 0:
            P = P
#            Rp = AverageDist(P, P_Summary, sample, dim)
            PF = np.asarray(P_fitness)
        else:
            P,P_fitness = CE_Online(sample,P_Summary,P,P_fitness,stdData,gamma,PreStd)
            PF = np.asarray(P_fitness)
        
#        if PFS:
#            [P,PF] = check(P_Summary,P,PF,PFS)
        
        P_Summary = ClusterSummary(P,PF,P_Summary,sample)
        
        PreStd,PFS = StoreInf(PF,PFS,PreStd,stdData)
        
#        [MinDist,ClusterIndice] = Cluster_Assign(sample,P)
    
    [MinDist,ClusterIndice] = Cluster_Assign(AccSample,P)
    
    num_S = 38
    
    FetchIndex = []
    UnlabeledIndex = []
    
    InterDist = squareform(pdist(P))
    
    for i in range(np.shape(P)[0]):
        tempcluster = np.where(ClusterIndice==(i))
        temp_interdist = InterDist[i,:]
        temp_rank = np.argsort(temp_interdist)
        temp_neigh1 = P[temp_rank[0],:]
        temp_neigh2 = P[temp_rank[1],:]
        
        d1 = []
        temp_dist1 = pdist(AccSample[tempcluster[0],:])
        temp_dist2 =  squareform(temp_dist1)

        for j in range(len(tempcluster[0])):
            knd1_ind = np.argsort(temp_dist2[j,:])
            knd1 = np.mean(temp_dist2[j,knd1_ind[:5]])
            d1.append(np.linalg.norm(AccSample[tempcluster[0][j],:]-P[i,:])/knd1)
#            d2.append(np.linalg.norm(AccSample[tempcluster[0][j],:]-P[i,:]))
        fetchSize = num_S * len(d1)/np.shape(AccSample)[0]
        sortIndex1 = np.argsort(d1)
        fet1 = tempcluster[0][sortIndex1[:round(fetchSize*0.5)]]
        fet1 = fet1.astype(int)
        
        fil_index = sortIndex1[-round(len(d1)/2):]
        d2 = []
        
        for k in range(len(fil_index)):
            knd2_ind = np.argsort(temp_dist2[j,:])
            knd2 = np.mean(temp_dist2[j,knd2_ind[:5]])
            temp_d1 = np.linalg.norm(AccSample[tempcluster[0][fil_index[k]],:]-temp_neigh1) + np.linalg.norm(AccSample[tempcluster[0][fil_index[k]],:]-temp_neigh2)
            temp_d2 = abs(np.linalg.norm(AccSample[tempcluster[0][fil_index[k]],:]-temp_neigh1) - np.linalg.norm(AccSample[tempcluster[0][fil_index[k]],:]-temp_neigh2))
#            d2.append(temp_d2/temp_d1)
#            d2.append(temp_d2/knd2)
            d2.append(temp_d2)
        
        sortIndex2 = np.argsort(d2)
        
        candidate_fet2 = fil_index[sortIndex2[:round(fetchSize*0.8)]]
        sum_dist = []
        for ii in range(len(candidate_fet2)):
            candidate_d1=np.linalg.norm(AccSample[tempcluster[0][candidate_fet2[ii]],:]-temp_neigh1)
            candidate_d2=np.linalg.norm(AccSample[tempcluster[0][candidate_fet2[ii]],:]-temp_neigh1)
            sum_dist.append(candidate_d1+candidate_d2)
        
        sortIndex3 = np.argsort(sum_dist)
        
#        fet2 = tempcluster[0][sortIndex[round(fetchSize/2):round(fetchSize)]]
#        fet2 = tempcluster[0][fil_index[sortIndex2[:round(fetchSize*0.5)]]]
        fet2 = tempcluster[0][candidate_fet2[sortIndex3[:round(fetchSize*0.5)]]]
        fet2 = fet2.astype(int)
#        FetchIndex = np.append(FetchIndex,tempcluster[0][sortIndex[:round(fetchSize*1)]])
        FetchIndex = np.append(FetchIndex,fet1)
        FetchIndex = np.append(FetchIndex,fet2)  
#        UnlabeledIndex = np.append(UnlabeledIndex,tempcluster[0][round(fetchSize):])
        UnlabeledIndex = np.append(UnlabeledIndex,tempcluster[0][sortIndex1[round(fetchSize*0.5):round(len(d1)/2)]])
        UnlabeledIndex = np.append(UnlabeledIndex,tempcluster[0][fil_index[sortIndex2[round(fetchSize*0.5):len(d2)]]])

    FetchIndex = FetchIndex.astype(int)
    UnlabeledIndex = UnlabeledIndex.astype(int)

    sample_Fetch = AccSample[FetchIndex][:]
    sample_Unlabeled = AccSample[UnlabeledIndex][:]
    
    label_Fetch = label[FetchIndex]
    label_Unlabeled = label[UnlabeledIndex]
    
#    reprsentative_power = Post_Sampling(sample_Fetch,label[FetchIndex],5)
#    
#    sort_idx2 = np.argsort(reprsentative_power)
#    
#    reverse_idx = sort_idx2[::-1]
#    
#    new_fetchX = sample_Fetch[reverse_idx[:round(len(FetchIndex)*0.85)]][:]
#    new_fetchY = label_Fetch[reverse_idx[:round(len(FetchIndex)*0.85)]]
    
    new_fetchX = sample_Fetch
    new_fetchY = label_Fetch

    clf1 = KNeighborsClassifier(n_neighbors=3)
    clf2 = LR()
    clf3 = LinearSVC()

    clf1 = clf1.fit(new_fetchX, new_fetchY)
    clf2 = clf2.fit(new_fetchX, new_fetchY)
    clf3 = clf3.fit(new_fetchX, new_fetchY)

    pred_label1 = clf1.predict(sample_Unlabeled)
    pred_label2 = clf2.predict(sample_Unlabeled)
    pred_label3 = clf3.predict(sample_Unlabeled)

    Acc1 = clf1.score(sample_Unlabeled, label_Unlabeled)
    Acc2 = clf2.score(sample_Unlabeled, label_Unlabeled)
    Acc3 = clf3.score(sample_Unlabeled, label_Unlabeled)

    F1_1 = f1_score(label_Unlabeled, pred_label1, average='macro')
    F1_2 = f1_score(label_Unlabeled, pred_label2, average='macro')
    F1_3 = f1_score(label_Unlabeled, pred_label3, average='macro')

    P1_1 = precision_score(label_Unlabeled, pred_label1, average='macro')
    P1_2 = precision_score(label_Unlabeled, pred_label2, average='macro')
    P1_3 = precision_score(label_Unlabeled, pred_label3, average='macro')

    R1_1 = recall_score(label_Unlabeled, pred_label1, average='macro')
    R1_2 = recall_score(label_Unlabeled, pred_label2, average='macro')
    R1_3 = recall_score(label_Unlabeled, pred_label3, average='macro')

    # AUC1 = auc(label_Unlabeled, pred_label1)
    # AUC2 = auc(label_Unlabeled, pred_label2)
    # AUC3 = auc(label_Unlabeled, pred_label3)

    print('Accuracy:', Acc1, Acc2, Acc3)

    print('F-measures:', F1_1, F1_2, F1_3)

    print('Precision:', P1_1, P1_2, P1_3)

    print('Recall:', R1_1, R1_2, R1_3)

    # print('AUC:', AUC1, AUC2, AUC3)

    end = time.time()
    ExecutionTime = end - start
    print('The total Extection Time: ' + str(ExecutionTime))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 17:15:52 2018

@author: r17baena
"""
import numpy as np
import math as m
import random
from scipy.spatial import distance
filename = "dat.npz"
#source_tensor = H
#np.savez(filename,data=source_tensor)

def spread_users_one_antenna(RRH_pos,Radius,K):
    user_pos = np.zeros((K,2))

    # magic numbers!
    a = m.sqrt(3)
    b = a*Radius
    x = 2*Radius*(np.random.rand(K,1)-0.5);            # X & Y relative to the eNode's center 
    y = 2* Radius*a*(np.random.rand(K,1)-0.5)
    user_pos = np.zeros((K,2))
    user_pos[:,0] = x[:,0]
    user_pos[:,1] = y[:,0]
    R=1
    # odd is the distance to the closest antenna
    odd = np.zeros((R,K))
    for t in range(K):
        odd[:,t] = m.sqrt((user_pos[t,0]-RRH_pos[0])**2 +(user_pos[t,1]-RRH_pos[1])**2 )

    for t in range(K):
        # while user is badly spread (too far or too close), regenerate data
        while  (abs(user_pos[t,0])*m.sqrt(3)+ abs(user_pos[t,1]) >b ) or ( odd [0,t] < 5):
            user_pos[t,0] = 2*Radius*(random.random()-0.5)
            user_pos[t,1] = 0.5*Radius*a*(random.random()-0.5)
            odd[:,t] = (m.sqrt((user_pos[t,0]-RRH_pos[0])**2 +(user_pos[t,1]-RRH_pos[1])**2 ))

    # return location of users
    return user_pos


def channel(K,N,N_time):     #K users,N subcarriers, N_times slot.
    R=500                    #radius of the cell
    H=np.zeros((N_time,K,N)) #H[t,k,n] is the channel gain at time it on subband n for user k
    B=10 #
    chan_Trms = 500;        #Root mean square delay spread du canal en nanoseconde
    chan_v = 32          # longueur du canal en durée symbole ??
    chan_path = 10        # 10 paths (multipath channel)
    RRH_pos=np.array([0,0])
    for t in range(N_time):
        user_pos = spread_users_one_antenna(RRH_pos,R,K)
        for k in range(K):
            path = np.sort(np.vstack([np.array([0]), np.floor(np.random.rand(chan_path-1,1)*chan_v)]))
            h = (np.random.randn(chan_path,1)+1j*np.random.randn(chan_path,1))*np.exp(-path*(1000/B)/chan_Trms/2) 
            h = h/np.linalg.norm(h) #normalisation du canal
            H[t, k, :] = np.hstack(abs(np.fft.fft(h,n=N,axis=0)))
        #Large scalefading depending on the distance user- antenna
        
            d = distance.euclidean(user_pos[k,:],RRH_pos)
            
            path_loss_db=128.1+37.6*m.log10(d/1000)+ np.random.normal(0,8)
            path_loss=10**(-path_loss_db/10)
            H[t, k, :]= H[t, k, :]*m.sqrt(path_loss)
            
    return H
    
def PF_scheduler(H,P_total):
  #A_t[k,s] timeslot,user,nsubcarrier
#H_T: Channel matrix of size K*S.
#Bc: Bandwidth per subband
#Tk_t: average throughput achieved so far 
    N_time=np.size(H,0)  #nb timeslot
    K=np.size(H,1) #nb user
    S=np.size(H,2) #nb subcarrier
    N0 = 4e-21
    Bc=10E6/S
    tc=20
    P=P_total/S
    A=np.zeros((N_time,K,S))#allocation matrice
    R_possible=np.zeros((N_time,K,S))#possible data rate 
    Tk=np.zeros((N_time,K))
    pf=np.zeros((N_time,K,S))
    R=np.zeros((N_time,K))
    R_possible=gains_to_datarate(H,P_total)
    for t in range(N_time):
        
        for s in range(S):
            for k in range(K):
                if float(abs(Tk[t,k]))>np.finfo(float).eps:
                    pf[t,k,s]=R_possible[t,k,s]/Tk[t,k]
                else:
                    pf[t,k,s]=R_possible[t,k,s]
            k_b=np.argmax(pf[t,:,s])#seeking for user who max pf forsubcarrier s
            A[t,k_b,s]=1
            if t>1:
                Tk[t,k_b]=Tk[t-1,k_b]*(1-1/tc)+R_possible[t,k_b,s]/tc
            else:
                Tk[t,k_b]=R_possible[t,k_b,s]/tc
            R[t,k_b]+=R_possible[t,k_b,s]
        Tk[t,:]=np.zeros(K)
        if t>1:
            Tk[t,:]=Tk[t-1,:]*(1-1/tc)
        Tk[t,:]=Tk[t,:]+R[t,:]/tc
    return Tk

def gains_to_datarate(H, P_total):
    N = H.shape[1]
    P = P_total/N
    Bc = 1e6 / N
    N0 = 4e-21

    return Bc*np.log2(1+P*np.square(H)/(N0*Bc))

def generate_dataset(K, N, N_time, P_total, N_examples):
    x = np.zeros((N_examples, K * N * N_time))
    y=np.zeros((N_examples, K * N_time))#label
    r=np.zeros((N_examples,K* N_time*N))
    for i in range(N_examples):
        r[i,:]=gains_to_datarate(channel(K, N, N_time),P_total).reshape(K*N*N_time)
        x[i,:] = gains_to_datarate(channel(K, N, N_time), P_total).reshape(K*N*N_time)
        y[i,:]=PF_scheduler(channel(K, N, N_time),P_total).reshape(K*N_time)
    return x,r,y

x,r,l=generate_dataset(5, 7, 12, 10, 100)
filename = "data_input.npz"
source_tensor = x,r
np.savez(filename,data=source_tensor)
filename = "data_label.npz"
source_tensor = l
np.savez(filename,data=source_tensor)



def Tk_Network_Output(R,A):
    N_time=np.size(R,0)  #nb timeslot
    K=np.size(R,1) 
    Tk=np.zeros((N_time,K))
    tc=20
    for t in range(N_time):
        A_t=np.transpose(A[t,:,:])
        Rtot=np.diagonal(np.matmul(R[t,:,:],A_t))
        if t>1:
            Tk[t,:]=Tk[t-1,:]*(1-1/tc)
        Tk[t,:]=Tk[t,:]+Rtot/tc
    return Tk
    
    
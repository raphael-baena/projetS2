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



def fft(h,N):
    fft=np.fft.fft(h,axis=0)
    n=np.size(fft)
    if n>N:
        fft=fft[:N]
    if n<N:
        fft=np.vstack([fft,np.vstack(np.zeros(N-n))])
    return fft


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

            H[t, k, :] = np.hstack(abs(fft(h,N)))
        #Large scalefading depending on the distance user- antenna
        
            d = distance.euclidean(user_pos[k,:],RRH_pos)
            
            path_loss_db=128.1+37.6*m.log10(d/1000)+ np.random.normal(0,8)
            path_loss=10**(-path_loss_db/10)
            H[t, k, :]= H[t, k, :]*m.sqrt(path_loss)
            
    return H
    
def PF_scheduler(H_t,A_t,P_total,N0,Tk_t):
  #A_t[k,s] timeslot,user,nsubcarrier
#H_T: Channel matrix of size K*S.
#Bc: Bandwidth per subband
#Tk_t: average throughput achieved so far 

    K=np.size(H_t,0)
    S=np.size(H_t,1)

    Bc=10E6/S


    tc=20

    P=P_total/S


    for k in range(K):
        R=0
        for s in range(S):
            R=Bc*m.log2(1+P*H_t[k,s]**2/(N0*Bc))*A_t[k,s]
            R+=R
        Tk_t[k]=Tk_t[k]*(1-1/tc)+R/tc
    return Tk_t

def gains_to_datarate(H, P_total):
    N = H.shape[1]
    P = P_total/N
    Bc = 1e6 / N
    N0 = 4e-21

    return Bc*np.log2(1+P*np.square(H)/(N0*Bc))

def generate_dataset(K, N, N_time, P_total, N_examples):
    x = np.zeros((N_examples, K * N * N_time))
    for i in range(N_examples):
        x[i,:] = gains_to_datarate(channel(K, N, N_time), P_total).reshape(K*N*N_time)
    return x

generate_dataset(5, 7, 10, 1, 10)

from numpy import random,zeros,argpartition,diag,ones,inf,roots,tensordot,eye,ceil,argmax,\
    array,fill_diagonal,outer,copy,emath,trace,sign,maximum,linspace,where,savez
import time
from scipy.linalg import norm,eigh

from solve_maxcut import bcm_bm,cg_bm

import matplotlib.pyplot as pylot
#%% test code

# T = 50
T = 1

f_val_cg_total = zeros((4,T,400))
time_elapsed_cg_total = zeros((4,T,400))

random.seed(0)

for t in range(T):
    print("retry = ",t)
    
    # n = 5000
    # maxtime = 3
    
    n = 20000
    maxtime = 20
    
    # random
    A2 = random.randn(n,n)/n
    # A2 = random.randn(n,n)
    
    # shift = 50/n
    shift = 0
    A2 = A2 + A2.T + eye(n) * shift
    
    min_eig,min_vec = eigh(A2,subset_by_index = [0,0])
    
    c = ones(n)
    
    # k = 50
    k = int(ceil((2 * n) ** 0.5) * 1)
    B0 = random.randn(k,n)
    B0 = B0/norm(B0,axis = 0) * c ** 0.5
    
    # gamma = 1e-3
    gamma = 0.1
    # Conditional Gradient (CG) (parallel BCM for linear)
    B_cg1,f_val_cg1,time_elapsed_cg1 = cg_bm(A2,c ** 0.5,B0.copy(),maxtime = maxtime,maxiter = 2000)
    
    B_cg2,f_val_cg2,time_elapsed_cg2 = cg_bm(A2 + eye(n) * 20/n,c ** 0.5,B0.copy(),maxtime = maxtime,maxiter = 2000)
    
    B_cg3,f_val_cg3,time_elapsed_cg3 = cg_bm(A2 + eye(n) * 50/n,c ** 0.5,B0.copy(),maxtime = maxtime,maxiter = 2000)
    
    B_cg4,f_val_cg4,time_elapsed_cg4 = cg_bm(A2 + eye(n) * (-min_eig + gamma),c ** 0.5,B0.copy(),maxtime = maxtime,maxiter = 2000)
    
    f_val_cg2 -= 20
    f_val_cg3 -= 50
    f_val_cg4 -= (-min_eig + gamma) * n
    

    
    BM0 = B0.dot(A2)
    f0 = tensordot(B0,BM0)
    
    f_val_cg1[0] = f0
    f_val_cg2[0] = f0
    f_val_cg3[0] = f0
    f_val_cg4[0] = f0
    
    # f_val_cg_total[0,t,:f_val_cg1.shape[0]] = f_val_cg1
    # f_val_cg_total[1,t,:f_val_cg2.shape[0]] = f_val_cg2
    # f_val_cg_total[2,t,:f_val_cg3.shape[0]] = f_val_cg3
    # f_val_cg_total[3,t,:f_val_cg4.shape[0]] = f_val_cg4
    
    # time_elapsed_cg_total[0,t,:time_elapsed_cg1.shape[0]] =  time_elapsed_cg1
    # time_elapsed_cg_total[1,t,:time_elapsed_cg2.shape[0]] =  time_elapsed_cg2
    # time_elapsed_cg_total[2,t,:time_elapsed_cg3.shape[0]] =  time_elapsed_cg3
    # time_elapsed_cg_total[3,t,:time_elapsed_cg4.shape[0]] =  time_elapsed_cg4
    
    
#%%

pylot.ion()  # Turn on interactive mode
fig = pylot.figure(0)
pylot.plot(time_elapsed_cg1,f_val_cg1,label = "CG1",color = "black")
pylot.plot(time_elapsed_cg2,f_val_cg2,label = "CG2",color = "yellow")
pylot.plot(time_elapsed_cg3,f_val_cg3,label = "CG3",color = "red")
pylot.plot(time_elapsed_cg4,f_val_cg4,label = "CG4",color = "green")

pylot.xlabel("Time Elapsed")
pylot.ylabel("Function Value")
pylot.legend()

# pylot.show(block=True)  # Keep the window open

#%%

savez("comparsion_cg_sigma_T_"+str(T) +"_n_" + str(n),f_val_cg_total = f_val_cg_total,\
      time_elapsed_cg_total = time_elapsed_cg_total)
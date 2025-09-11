from numpy import random,zeros,argpartition,diag,ones,inf,roots,tensordot,eye,ceil,argmax,\
    array,fill_diagonal,outer,copy,emath,trace,sign,maximum,linspace,where
from numpy.linalg import norm
import time 

def bcm_bm(M,w,B,maxtime,maxiter = 200):
    m,n = M.shape
    
    diag_M = diag(M).copy()
    fill_diagonal(M,0)
    
    f_val = zeros(maxiter +1)
    time_elapsed = zeros(maxiter+1)
    for t in range(1,maxiter+1):
        t0 = time.time()
        for i in range(n):
            c = M[:,[i]]
            g = B.dot(c)
            # B[:,[i]] = g/norm(g) * w[i]
            B[:,[i]] = g/norm(g)
        t1 = time.time()
        
        BM = B.dot(M)
        f_val[t] = tensordot(B,BM) + w.dot(diag_M)
        time_elapsed[t] = time_elapsed[t-1] + t1-t0
        if time_elapsed[t] > maxtime:
            break
    f_val = f_val[:t+1]
    time_elapsed = time_elapsed[:t+1]
    
    
    fill_diagonal(M,diag_M)
    # Z = B.T.dot(B)
    # primals = ((Z * (Z.dot(M))).sum(axis = 0))/norm(Z,axis = 0) ** 2
    # return B,Z,primals,f_val,time_elapsed
    return B,f_val,time_elapsed

def cg_bm(M,w,B,maxtime,maxiter = 500):
    f_val = zeros(maxiter +1)
    time_elapsed = zeros(maxiter+1)
    for t in range(1,maxiter+1):
        t0 = time.time()
        B = B.dot(M)
        # B = B/norm(B,axis = 0) * w
        B = B/norm(B,axis = 0)
        t1 = time.time()
        
        BM = B.dot(M)
        f_val[t] = tensordot(B,BM)        
        time_elapsed[t] = time_elapsed[t-1] + t1-t0
        if time_elapsed[t] > maxtime:
            break
    
    f_val = f_val[:t+1]
    time_elapsed = time_elapsed[:t+1]
    
    # Z = B.T.dot(B)
    # primals = ((Z * (Z.dot(M))).sum(axis = 0))/norm(Z,axis = 0) ** 2
    # return B,Z,primals,f_val,time_elapsed
    return B,f_val,time_elapsed
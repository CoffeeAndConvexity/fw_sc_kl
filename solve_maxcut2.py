from numpy import random,zeros,argpartition,diag,ones,inf,roots,tensordot,eye,ceil,argmax,\
    array,fill_diagonal,outer,copy,emath,trace,sign,maximum,linspace,where
from numpy.linalg import norm
import time
import numpy as np

import pymanopt
from pymanopt.manifolds import Oblique
from pymanopt.optimizers import TrustRegions

# import matplotlib.pyplot as pylot

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


def cg_bm32(M,M64,w,B,maxtime,maxiter = 500):
    f_val = zeros(maxiter +1)
    time_elapsed = zeros(maxiter+1)
    for t in range(1,maxiter+1):
        t0 = time.time()
        B = B.dot(M)
        # B = B/norm(B,axis = 0) * w
        B = B/norm(B,axis = 0)
        t1 = time.time()
        
        BM = B.dot(M64)
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

# def cg_bm2(M,w,B,maxtime,maxiter = 500):
#     f_val = zeros(maxiter +1)
#     time_elapsed = zeros(maxiter+1)
#     for t in range(1,maxiter+1):
#         t0 = time.time()
#         # B = B.dot(M)
#         B = M.dot(B)
#         # B = B/norm(B,axis = 0) * w
#         B = B/norm(B,axis = 0)
#         t1 = time.time()
        
#         BM = B.dot(M)
#         f_val[t] = tensordot(B,BM)        
#         time_elapsed[t] = time_elapsed[t-1] + t1-t0
#         if time_elapsed[t] > maxtime:
#             break
    
#     f_val = f_val[:t+1]
#     time_elapsed = time_elapsed[:t+1]
    
#     # Z = B.T.dot(B)
#     # primals = ((Z * (Z.dot(M))).sum(axis = 0))/norm(Z,axis = 0) ** 2
#     # return B,Z,primals,f_val,time_elapsed
#     return B,f_val,time_elapsed

def manopt(A,B,maxtime):
    r,n = B.shape
    print("n",n,"r",r)
    manifold = Oblique(r,n)
    @pymanopt.function.numpy(manifold)
    def cost(X):
        return -trace(X @ A @ X.T)

    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(X):
        return -2 * X @ A

    @pymanopt.function.numpy(manifold)
    def euclidean_hessian(X, H):
        return -2 * H @ A
    
    problem = pymanopt.Problem(
        manifold,
        cost,
        euclidean_gradient=euclidean_gradient,
        euclidean_hessian=euclidean_hessian,
    )
    
    optimizer = TrustRegions(verbosity=2,max_time = maxtime)
    optimizer._log_verbosity = 0
    solution = optimizer.run(problem,initial_point= B)
    # solution = optimizer.run(problem)
    return problem,optimizer,solution

#%%

random.seed(0)
n = 5000
k = 50
C = random.randn(n,n)/n

shift = 50/n
C = C + C.T + eye(n) * shift

B0 = random.randn(k,n)
B0 = B0/norm(B0,axis = 0)

C_32 = C.astype(np.float32)
B_32 = B0.astype(np.float32)

#%%

T = 5

# B1,f1,t1 = bcm_bm(C_32,ones(n),B_32.copy(),maxtime =T,maxiter = 4000)
B2,f2,t2 = cg_bm32(C_32,C,ones(n),B_32.copy(),maxtime =T,maxiter = 4000)
B3,f3,t3 = cg_bm(C,ones(n),B0.copy(),maxtime =T,maxiter = 4000)
p,o,s = manopt(C,B0.copy(),maxtime = T)
p32,o32,s32 = manopt(C_32,B_32.copy(),maxtime = T)

#%%
# print("bcm",f1[-1])
print("cg32",f2[-1])
print("cg64",f3[-1])
print("manopt64",s.cost)
print("manopt32",s32.cost)
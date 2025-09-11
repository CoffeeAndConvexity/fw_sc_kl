from numpy import zeros,argpartition,log2,ceil,count_nonzero,std,sort,random,array,where,eye,ones,diag,inf,\
    savez,load,arange,sign
import time
from scipy.sparse.linalg import cg
from scipy.linalg import solve,norm,eigh,cholesky
import cvxpy as cp

#%% generate instance

T = 200
S = [20,60]
increment = 2

# testing
# T = 10
# S = [50,51]
# increment = 2

algo_count = 3
recovery = zeros(((S[1]-S[0])//increment+1,T,algo_count))
wall = zeros(((S[1]-S[0])//increment+1,T,algo_count))
# fval = zeros(((S[1]-S[0])//increment+1,T,algo_count))
iter_count = zeros(((S[1]-S[0])//increment+1,T,algo_count))
xset = zeros(((S[1]-S[0])//increment+1,T,algo_count + 1),dtype = object)


# chosen = random.randint(0,1e6)
# # chosen = 229385
# random.seed(chosen)
random.seed(0)
eps = 0.1
retry = 10
convergence_criterion = 1e-3
convergence_criterion2 = convergence_criterion ** 2
for s in range(S[0],S[1]+1,2):
  for t in range(T):
      [m,n] = [100,256]
      A = random.randn(m,n)
      A = A/norm(A,axis = 0)
      
      x0 = zeros(n)
      support_set = random.choice(range(n),s,replace = True)
      magnify = 1
      x0[support_set] = random.randn(s) * magnify
      noise = 0.0
      b = A.dot(x0) + random.randn(m) * noise
      xset[(s - S[0])//increment,t,-1] = x0
      
      print("-----Sparsity",s,"-----Trial-----",t,"---------")
      #
      # l1 norm solution
      #
      x_l1 = cp.Variable(shape=n)
      constraints_l1 = [A @ x_l1 == b]
      obj_l1 = cp.Minimize(cp.norm(x_l1, 1))
      prob_l1 = cp.Problem(obj_l1, constraints_l1)
      prob_l1.solve(solver = cp.MOSEK)
      print("status: {}".format(prob_l1.status))
      print("l1")
      print("norm diff",norm(x_l1.value-x0))
      
      #
      # rwl1 original
      #
      print("rwl1")
      x_rwl1_old = x_l1.value
      W_rwl1 = cp.Parameter(shape=n, nonneg=True);
      W_rwl1.value = ones(n)/(abs(x_l1.value) + eps)
      x_rwl1 = cp.Variable(shape=n)
      constraints_rwl1 = [A @ x_rwl1 == b]
      obj_rwl1 = cp.Minimize(W_rwl1.T @ cp.abs(x_rwl1))
      prob_rwl1 = cp.Problem(obj_rwl1,constraints_rwl1)

      i = 0
      w = 0
      converged = False
      while not converged:
          i += 1
          prob_rwl1.solve(solver = cp.MOSEK)
          w += prob_rwl1._solve_time
          # print("status: {}".format(prob_rwl1.status))
          W_rwl1.value = ones(n)/(abs(x_rwl1.value) + eps)
          print("iter",i,"rwl1 norm diff",norm(x_rwl1.value-x0))
          if norm(x_rwl1.value - x_rwl1_old) <= convergence_criterion:
              converged = True
          else:
              x_rwl1_old = x_rwl1.value
      
      if norm(x_rwl1.value-x0) <= convergence_criterion:
          recovery[(s - S[0])//increment,t,0] = 1
          
      iter_count[(s - S[0])//increment,t,0] = i
      wall[(s - S[0])//increment,t,0] = w
      xset[(s - S[0])//increment,t,0] = x_rwl1.value
      
      #
      # rwl1 proximal
      #
      print("rwl1 prox") 
      lamda = 0.1
      mu = 1
      xp_rwl1_prox_old = x_l1.value.clip(min = 0)
      # xn_rwl1_prox_old = x_l1.value.clip(max = 0)
      xn_rwl1_prox_old = -1 * x_l1.value.clip(max = 0)
      W_rwl1_prox = cp.Parameter(shape=n, nonneg=True);
      W_rwl1_prox.value = ones(n)/(abs(x_l1.value) + eps)
      xp_prox = cp.Parameter(shape=n, nonneg=True);
      xp_prox.value = x_l1.value.clip(min = 0)
      xn_prox = cp.Parameter(shape=n, nonneg=True);
      xn_prox.value = -1 * x_l1.value.clip(max = 0)
      
      xp_rwl1_prox = cp.Variable(shape= n, nonneg = True)
      xn_rwl1_prox = cp.Variable(shape= n, nonneg = True)
      
      constraints_rwl1_prox = [A @ xp_rwl1_prox - A @ xn_rwl1_prox == b]
      obj_rwl1_prox = cp.Minimize(W_rwl1_prox.T @ (xp_rwl1_prox + xn_rwl1_prox) + \
          +lamda * cp.sum_squares(xp_rwl1_prox-xp_prox) + lamda * cp.sum_squares(xn_rwl1_prox-xn_prox))
      prob_rwl1_prox = cp.Problem(obj_rwl1_prox,constraints_rwl1_prox)
      
      i = 0
      w = 0
      converged = False
      while not converged:
          i += 1
          prob_rwl1_prox.solve(solver = cp.MOSEK)
          w += prob_rwl1_prox._solve_time
          # print("status: {}".format(prob_rwl1_prox.status))
          xp_prox.value = xp_rwl1_prox.value
          xn_prox.value = xn_rwl1_prox.value
          beta = mu * (xp_rwl1_prox.value + xn_rwl1_prox.value + eps) - W_rwl1_prox.value
          W_rwl1_prox.value = (-beta + (beta ** 2 + 4) ** 0.5)/2
          print("iter",i,"prox rwl1 norm diff",norm(xp_rwl1_prox.value-xn_rwl1_prox.value-x0))
          if norm(xp_rwl1_prox.value - xp_rwl1_prox_old) ** 2 + \
              norm(xn_rwl1_prox.value - xn_rwl1_prox_old) ** 2  <= convergence_criterion2:
              converged = True
          else:
              xp_rwl1_prox_old = xp_rwl1_prox.value
              xn_rwl1_prox_old = xn_rwl1_prox.value
              
      if norm(xp_rwl1_prox.value-xn_rwl1_prox.value-x0) <= convergence_criterion:
          recovery[(s - S[0])//increment,t,1] = 1
      
      iter_count[(s - S[0])//increment,t,1] = i
      wall[(s - S[0])//increment,t,1] = w
      xset[(s - S[0])//increment,t,1] = xp_rwl1_prox.value-xn_rwl1_prox.value
      
      #
      # rwl1*
      #
      print("rwl1 star")
      xp_rwl1_star_old = x_l1.value.clip(min = 0)
      # xn_rwl1_star_old = x_l1.value.clip(max = 0)
      xn_rwl1_star_old = -1 * x_l1.value.clip(max = 0)
      Wp_rwl1_star = cp.Parameter(shape=n, nonneg=True);
      Wp_rwl1_star.value = ones(n)/(x_l1.value.clip(min = 0) + eps)
      Wn_rwl1_star = cp.Parameter(shape=n, nonneg=True);
      Wn_rwl1_star.value = ones(n)/(-1 * x_l1.value.clip(max = 0) + eps)
      
      xp_rwl1_star = cp.Variable(shape= n, nonneg = True)
      xn_rwl1_star = cp.Variable(shape= n, nonneg = True)
      
      constraints_rwl1_star = [A @ xp_rwl1_star - A @ xn_rwl1_star == b]
      obj_rwl1_star = cp.Minimize(Wp_rwl1_star.T @ xp_rwl1_star + Wn_rwl1_star @ xn_rwl1_star )
      prob_rwl1_star = cp.Problem(obj_rwl1_star,constraints_rwl1_star)
      
      i = 0
      w = 0
      converged = False
      while not converged:
          i += 1
          prob_rwl1_star.solve(solver = cp.MOSEK)
          w += prob_rwl1_star._solve_time
          # print("status: {}".format(prob_rwl1_star.status))
          Wp_rwl1_star.value = ones(n)/(xp_rwl1_star.value + eps)
          Wn_rwl1_star.value = ones(n)/(xn_rwl1_star.value + eps)
          print("iter",i,"rwl1* norm diff",norm(xp_rwl1_star.value-xn_rwl1_star.value-x0))
          if norm(xp_rwl1_star.value - xp_rwl1_star_old) ** 2 + \
              norm(xn_rwl1_star.value - xn_rwl1_star_old) ** 2  <= convergence_criterion2:
              converged = True
          else:
              xp_rwl1_star_old = xp_rwl1_star.value
              xn_rwl1_star_old = xn_rwl1_star.value
      
      if norm(xp_rwl1_star.value-xn_rwl1_star.value-x0) <= convergence_criterion:
          recovery[(s - S[0])//increment,t,2] = 1
      
      iter_count[(s - S[0])//increment,t,2] = i
      wall[(s - S[0])//increment,t,2] = w
      xset[(s - S[0])//increment,t,2] = xp_rwl1_star.value-xn_rwl1_star.value
        

#%% save file

savez("rwl_comparison",recovery = recovery,wall = wall,iter_count = iter_count,xset = xset)
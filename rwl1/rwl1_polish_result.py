import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

data = np.load("rwl_comparison.npz",allow_pickle = True)

recovery = data["recovery"]
wall = data["wall"]
iter_count = data["iter_count"]
xset = data["xset"]

wall_per_iter = wall/iter_count

recovery_mean = recovery.mean(axis = 1)
wall_mean = wall.mean(axis = 1)
iter_count_mean = iter_count.mean(axis = 1)
wall_per_iter_mean = wall_per_iter.mean(axis = 1)

sparsity = np.arange(20,61,2)

#%%

print(np.linalg.norm(iter_count[:,:,0]-iter_count[:,:,2]))
print(np.linalg.norm(recovery[:,:,0]-recovery[:,:,2]))

r,c = np.where(recovery[:,:,0]-recovery[:,:,2] != 0)

#%% recovery

fig0 = plt.figure(0)

plt.plot(sparsity, recovery_mean.T[0], color='red',linestyle = "solid", linewidth=2, label="RWL1")
plt.plot(sparsity, recovery_mean.T[1], color='blue',linestyle = "dashdot", linewidth=2, label="RWL1 Prox")
plt.plot(sparsity, recovery_mean.T[2], color='green',marker = "*",markersize = 6, linewidth=2, label="RWL1 Split",linestyle = "dashed")

plt.xlabel("Cardinality s",fontsize = 18)
plt.ylabel("Recovery Rate",fontsize = 18)
plt.grid(True)
plt.legend()
plt.show()
plt.title("Sparse Signal Recovery",fontsize = 20)

# fig0.savefig("rwl1_recovery.png",format ="png")
# fig0.savefig("rwl1_recovery.eps",format ="eps")
fig0.savefig("rwl1_recovery.pdf",format ="pdf")

#%% wall time

fig1 = plt.figure(1)

plt.plot(sparsity, wall_mean.T[0], color='red',linestyle = "solid", linewidth=2, label="RWL1")
plt.plot(sparsity, wall_mean.T[1], color='blue',linestyle = "dashdot", linewidth=2, label="RWL1 Prox")
plt.plot(sparsity, wall_mean.T[2], color='green',marker = "*",markersize = 6, linewidth=2, label="RWL1 Split",linestyle = "dashed")

plt.xlabel("Cardinality s",fontsize = 18)
plt.ylabel("Wall Time",fontsize = 18)
plt.grid(True)
plt.legend()
plt.show()
plt.title("Computation Time",fontsize = 20)

# fig1.savefig("rwl1_wall.png",format ="png")
# fig1.savefig("rwl1_wall.eps",format ="eps")
fig1.savefig("rwl1_wall.pdf",format ="pdf")

#%% iter count

fig2 = plt.figure(2)

plt.plot(sparsity, iter_count_mean.T[0], color='red',linestyle = "solid", linewidth=2, label="RWL1")
plt.plot(sparsity, iter_count_mean.T[1], color='blue',linestyle = "dashdot", linewidth=2, label="RWL1 Prox")
plt.plot(sparsity, iter_count_mean.T[2], color='green',marker = "*",markersize = 6, linewidth=2, label="RWL1 Split",linestyle = "dashed")

plt.xlabel("Cardinality s",fontsize = 18)
plt.ylabel("Number of iterations",fontsize = 18)
plt.grid(True)
plt.legend()
plt.show()
plt.title("Prox/CG Iterations",fontsize = 20)

# fig2.savefig("rwl1_iter_count.png",format ="png")
# fig2.savefig("rwl1_iter_count.eps",format ="eps")
fig2.savefig("rwl1_iter_count.pdf",format ="pdf")

#%% wall time per cg iter


fig3 = plt.figure(3)

plt.plot(sparsity, wall_per_iter_mean.T[0], color='red',linestyle = "solid", linewidth=2, label="RWL1")
plt.plot(sparsity, wall_per_iter_mean.T[1], color='blue', linestyle = "dashdot",linewidth=2, label="RWL1 Prox")
plt.plot(sparsity, wall_per_iter_mean.T[2], color='green',marker = "*",markersize = 6, linewidth=2, label="RWL1 Split",linestyle = "dashed")

plt.xlabel("Cardinality s",fontsize = 18)
plt.ylabel("Wall time",fontsize = 18)
plt.grid(True)
plt.legend()
plt.show()
plt.title("Wall Time per CG/Prox Iter",fontsize = 20)

# fig3.savefig("rwl1_wall_per_cg.png",format ="png")
# fig3.savefig("rwl1_wall_per_cg.eps",format ="eps")
fig3.savefig("rwl1_wall_per_cg.pdf",format ="pdf")


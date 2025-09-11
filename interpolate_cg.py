import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

num_curves = 50
common_time = np.linspace(0, 60, 500)
data = np.load("comparsion_cg_sigma_T_"+str(num_curves)+"_n_20000.npz")

# test code
# num_curves = 10
# common_time = np.linspace(0, 1, 500)
# data = np.load("comparsion_cg_sigma_T_"+str(num_curves)+"_n_5000.npz")

f_val_cg_total = data["f_val_cg_total"]
time_elapsed_cg_total = data["time_elapsed_cg_total"]

time_stamps = time_elapsed_cg_total.argmax(axis = 2)

f0 = [f_val_cg_total[0][i,:time_stamps[0,i]] for i in range(num_curves)]
f1 = [f_val_cg_total[1][i,:time_stamps[1,i]] for i in range(num_curves)]
f2 = [f_val_cg_total[2][i,:time_stamps[2,i]] for i in range(num_curves)]
f3 = [f_val_cg_total[3][i,:time_stamps[3,i]] for i in range(num_curves)]

t0 = [time_elapsed_cg_total[0][i,:time_stamps[0,i]] for i in range(num_curves)]
t1 = [time_elapsed_cg_total[1][i,:time_stamps[1,i]] for i in range(num_curves)]
t2 = [time_elapsed_cg_total[2][i,:time_stamps[2,i]] for i in range(num_curves)]
t3 = [time_elapsed_cg_total[3][i,:time_stamps[3,i]] for i in range(num_curves)]

interpolated_curve_cg0 = np.array([
    interp1d(t, f, kind='cubic', fill_value="extrapolate")(common_time)
    for t, f in zip(t0, f0)
])

interpolated_curve_cg1 = np.array([
    interp1d(t, f, kind='cubic', fill_value="extrapolate")(common_time)
    for t, f in zip(t1, f1)
])

interpolated_curve_cg2 = np.array([
    interp1d(t, f, kind='cubic', fill_value="extrapolate")(common_time)
    for t, f in zip(t2, f2)
])

interpolated_curve_cg3 = np.array([
    interp1d(t, f, kind='cubic', fill_value="extrapolate")(common_time)
    for t, f in zip(t3, f3)
])


mean_curve_cg0 = np.mean(interpolated_curve_cg0, axis=0)
mean_curve_cg1 = np.mean(interpolated_curve_cg1, axis=0)
mean_curve_cg2 = np.mean(interpolated_curve_cg2, axis=0)
mean_curve_cg3 = np.mean(interpolated_curve_cg3, axis=0)

fig0 = plt.figure(0)

plt.plot(common_time, mean_curve_cg0,color = "black", linestyle = "solid", linewidth=2, label=r"$\sigma_1 = 0$")
plt.plot(common_time, mean_curve_cg1,color='orange',  linestyle = "dotted", linewidth=2, label=r"$\sigma_2 = 10^{-3}$")
plt.plot(common_time, mean_curve_cg2,color='red',     linestyle = "dashed", linewidth=2, label=r"$\sigma_3 = 25 \times 10^{-4}$")
plt.plot(common_time, mean_curve_cg3,color = "green", linestyle = "dashdot", linewidth=2, label=r"$\sigma_4 = -\lambda_{min}(A) + 10^{-1}$")


plt.xscale("log")

plt.xlabel("Wall Time (s)",fontsize = 16)
plt.ylabel("Objective Value",fontsize = 16)
plt.grid(True)
plt.legend()
plt.show()
plt.title(r"Effect of Choice of $\sigma$",fontsize = 18)

plt.tight_layout()  # Automatically adjusts all margins

# fig0.savefig("cg_sigma_comparison_T_"+str(num_curves)+".png",format ="png")
fig0.savefig("cg_sigma_comparison_T_"+str(num_curves)+".eps",format ="eps")
fig0.savefig("cg_sigma_comparison_T_"+str(num_curves)+".pdf",format ="pdf")

# fig0.savefig("cg_sigma_comparison_T_"+str(num_curves)+".eps",format ="eps",bbox_inches='tight', pad_inches=0.05)
# fig0.savefig("cg_sigma_comparison_T_"+str(num_curves)+".pdf",format ="pdf",bbox_inches='tight', pad_inches=0.05)


ci = 1
plt.xlim((55,60))
plt.ylim((mean_curve_cg1[-1]-ci,mean_curve_cg1[-1]+ci))

plt.tight_layout()  # Automatically adjusts all margins

# fig0.savefig("cg_sigma_comparison_T_"+str(num_curves)+"_last_second.png",format ="png")
fig0.savefig("cg_sigma_comparison_T_"+str(num_curves)+"_last_second.eps",format ="eps")
fig0.savefig("cg_sigma_comparison_T_"+str(num_curves)+"_last_second.pdf",format ="pdf")

# fig0.savefig("cg_sigma_comparison_T_"+str(num_curves)+"_last_second.eps",format ="eps",bbox_inches='tight', pad_inches=0.05)
# fig0.savefig("cg_sigma_comparison_T_"+str(num_curves)+"_last_second.pdf",format ="pdf",bbox_inches='tight', pad_inches=0.05)

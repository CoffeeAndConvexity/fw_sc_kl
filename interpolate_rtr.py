import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd

num_curves = 3
common_time = np.linspace(0, 60, 500)
# common_time = np.linspace(0, 2, 500)

initial = 50 * np.ones((50,1))
df_bcm = pd.read_csv('f_bcm.csv', header=None)
df_rgd = pd.read_csv('f_rgd.csv', header=None)
df_rtr = pd.read_csv('f_rtr.csv', header=None)
df_cg = pd.read_csv('f_cg.csv', header=None)
df_cg2 = pd.read_csv('f_cg2.csv', header=None)

# df_bcm = df_bcm.subtract(df_bcm.iloc[:,0],axis = 0)
# df_rgd = df_rgd.subtract(df_rgd.iloc[:,0],axis = 0)
# df_rtr = df_rtr.subtract(df_rtr.iloc[:,0],axis = 0)
# df_cg = df_cg.subtract(df_cg.iloc[:,0],axis = 0)
# df_cg2 = df_cg2.subtract(df_cg2.iloc[:,0],axis = 0)

df_bcm = df_bcm.subtract(initial,axis = 0)
df_rgd = df_rgd.subtract(initial,axis = 0)
df_rtr = df_rtr.subtract(initial,axis = 0)
df_cg = df_cg.subtract(initial,axis = 0)
df_cg2 = df_cg2.subtract(initial,axis = 0)

f_bcm = df_bcm.apply(lambda row: row.dropna().tolist(), axis=1).tolist()
f_rgd = df_rgd.apply(lambda row: row.dropna().tolist(), axis=1).tolist()
f_rtr = df_rtr.apply(lambda row: row.dropna().tolist(), axis=1).tolist()
f_cg = df_cg.apply(lambda row: row.dropna().tolist(), axis=1).tolist()
f_cg2 = df_cg2.apply(lambda row: row.dropna().tolist(), axis=1).tolist()


dt_bcm = pd.read_csv('telapsed_bcm.csv', header=None)
dt_rgd = pd.read_csv('telapsed_rgd.csv', header=None)
dt_rtr = pd.read_csv('telapsed_rtr.csv', header=None)
dt_cg = pd.read_csv('telapsed_cg.csv', header=None)
dt_cg2 = pd.read_csv('telapsed_cg2.csv', header=None)

dt_bcm[dt_bcm > 60] = 60
dt_rgd[dt_rgd > 60] = 60
dt_rtr[dt_rtr > 60] = 60
dt_cg[dt_cg > 60] = 60
dt_cg2[dt_cg2 > 60] = 60

t_bcm = dt_bcm.apply(lambda row: row.dropna().tolist(), axis=1).tolist()
t_rgd = dt_rgd.apply(lambda row: row.dropna().tolist(), axis=1).tolist()
t_rtr = dt_rtr.apply(lambda row: row.dropna().tolist(), axis=1).tolist()
t_cg = dt_cg.apply(lambda row: row.dropna().tolist(), axis=1).tolist()
t_cg2 = dt_cg2.apply(lambda row: row.dropna().tolist(), axis=1).tolist()


interpolation = "linear"


interpolated_curve_bcm = np.array([
    interp1d(t, f, kind=interpolation, fill_value="extrapolate")(common_time)
    for t, f in zip(t_bcm, f_bcm)
])

interpolated_curve_rgd = np.array([
    interp1d(t, f, kind=interpolation, fill_value="extrapolate")(common_time)
    for t, f in zip(t_rgd, f_rgd)
])

interpolated_curve_rtr = np.array([
    interp1d(t, f, kind=interpolation, fill_value="extrapolate")(common_time)
    for t, f in zip(t_rtr, f_rtr)
])

interpolated_curve_cg = np.array([
    interp1d(t, f, kind=interpolation, fill_value="extrapolate")(common_time)
    for t, f in zip(t_cg, f_cg)
])

interpolated_curve_cg2 = np.array([
    interp1d(t, f, kind=interpolation, fill_value="extrapolate")(common_time)
    for t, f in zip(t_cg2, f_cg2)
])


mean_curve_bcm = np.mean(interpolated_curve_bcm, axis=0)
mean_curve_rgd = np.mean(interpolated_curve_rgd, axis=0)
mean_curve_rtr = np.mean(interpolated_curve_rtr, axis=0)
mean_curve_cg = np.mean(interpolated_curve_cg, axis=0)
mean_curve_cg2 = np.mean(interpolated_curve_cg2, axis=0)

fig0 = plt.figure(0)

plt.plot(common_time, mean_curve_bcm,color = "blue", linestyle = "solid", linewidth=2, label="BCM")
plt.plot(common_time, mean_curve_rgd,color='orange',   linewidth=2,dashes = (3, 5, 1, 2), label="RGD")
plt.plot(common_time, mean_curve_rtr,color='red',     linestyle = "dashed", linewidth=2, label="RTR")
plt.plot(common_time, mean_curve_cg,color = "green", linestyle = "dashdot", linewidth=2, label="GFW")
# plt.plot(common_time, mean_curve_cg2,color = "purple", linestyle = "densely dotted", linewidth=2, label="CG + RTR")
plt.plot(common_time, mean_curve_cg2,color = "purple", linewidth=2, label="GFW & RTR", dashes = (1,1))

# plt.yscale('log')
plt.xscale("log")

plt.xlabel("Wall Time (s)",fontsize = 16)
plt.ylabel("Objective Value",fontsize = 16)
plt.grid(True)
plt.legend()
plt.show()
plt.title(r"Performance Against Time",fontsize = 18)

plt.tight_layout()  # Automatically adjusts all margins

# fig0.savefig("cg_rtr_comparison_T_"+str(num_curves)+".png",format ="png")
fig0.savefig("cg_rtr_comparison_T_"+str(num_curves)+".eps",format ="eps")
fig0.savefig("cg_rtr_comparison_T_"+str(num_curves)+".pdf",format ="pdf")

# fig0.savefig("cg_rtr_comparison_T_"+str(num_curves)+".eps",format ="eps",bbox_inches='tight', pad_inches=0.05)
# fig0.savefig("cg_rtr_comparison_T_"+str(num_curves)+".pdf",format ="pdf",bbox_inches='tight', pad_inches=0.05)


# plt.xlim((6,8))


ci = 1
plt.xlim((55,60))
plt.ylim((mean_curve_rtr[-1]-ci,mean_curve_rtr[-1]+ci))

plt.tight_layout()  # Automatically adjusts all margins

# # fig0.savefig("cg_rtr_comparison_T_"+str(num_curves)+"_last_second.png",format ="png")
fig0.savefig("cg_rtr_comparison_T_"+str(num_curves)+"_last_second.eps",format ="eps")
fig0.savefig("cg_rtr_comparison_T_"+str(num_curves)+"_last_second.pdf",format ="pdf")

# fig0.savefig("cg_rtr_comparison_T_"+str(num_curves)+"_last_second.eps",format ="eps",bbox_inches='tight', pad_inches=0.05)
# fig0.savefig("cg_rtr_comparison_T_"+str(num_curves)+"_last_second.pdf",format ="pdf",bbox_inches='tight', pad_inches=0.05)
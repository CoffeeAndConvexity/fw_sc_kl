import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd

num_curves = 50
common_time = np.linspace(0, 60, 500)
# common_time = np.linspace(0, 2, 500)

initial = 50 * np.ones((50,1))
initial = 0 * np.ones((50,1))

df_bcm = pd.read_csv('f_bcm.csv', header=None)
df_bcm2 = pd.read_csv('f_bcm2.csv', header=None)
df_rgd = pd.read_csv('f_rgd.csv', header=None)
df_rtr = pd.read_csv('f_rtr.csv', header=None)
df_cg = pd.read_csv('f_cg.csv', header=None)
df_cg2 = pd.read_csv('f_cg2.csv', header=None)
df_admm = pd.read_csv('f_admm.csv', header=None)
df_cgal1 = pd.read_csv('f_cgal1.csv', header=None)
df_cgal2 = pd.read_csv('f_cgal2.csv', header=None)
df_lorads = pd.read_csv('f_lorads.csv', header=None)

# df_bcm = df_bcm.subtract(df_bcm.iloc[:,0],axis = 0)
# df_rgd = df_rgd.subtract(df_rgd.iloc[:,0],axis = 0)
# df_rtr = df_rtr.subtract(df_rtr.iloc[:,0],axis = 0)
# df_cg = df_cg.subtract(df_cg.iloc[:,0],axis = 0)
# df_cg2 = df_cg2.subtract(df_cg2.iloc[:,0],axis = 0)

df_bcm = df_bcm.subtract(initial,axis = 0)
df_bcm2 = df_bcm2.subtract(initial,axis = 0)
df_rgd = df_rgd.subtract(initial,axis = 0)
df_rtr = df_rtr.subtract(initial,axis = 0)
df_cg = df_cg.subtract(initial,axis = 0)
df_cg2 = df_cg2.subtract(initial,axis = 0)
df_admm = df_admm.subtract(initial,axis = 0)
df_cgal1 = df_cgal1.subtract(initial,axis = 0)
df_cgal2 = df_cgal2.subtract(initial,axis = 0)
df_lorads = df_lorads.subtract(initial,axis = 0)

f_bcm = df_bcm.apply(lambda row: row.dropna().tolist(), axis=1).tolist()
f_bcm2 = df_bcm2.apply(lambda row: row.dropna().tolist(), axis=1).tolist()
f_rgd = df_rgd.apply(lambda row: row.dropna().tolist(), axis=1).tolist()
f_rtr = df_rtr.apply(lambda row: row.dropna().tolist(), axis=1).tolist()
f_cg = df_cg.apply(lambda row: row.dropna().tolist(), axis=1).tolist()
f_cg2 = df_cg2.apply(lambda row: row.dropna().tolist(), axis=1).tolist()
f_admm = df_admm.apply(lambda row: row.dropna().tolist(), axis=1).tolist()
f_cgal1 = df_cgal1.apply(lambda row: row.dropna().tolist(), axis=1).tolist()
f_cgal2 = df_cgal2.apply(lambda row: row.dropna().tolist(), axis=1).tolist()
f_lorads = df_lorads.apply(lambda row: row.dropna().tolist(), axis=1).tolist()

dt_bcm = pd.read_csv('telapsed_bcm.csv', header=None)
dt_bcm2 = pd.read_csv('telapsed_bcm2.csv', header=None)
dt_rgd = pd.read_csv('telapsed_rgd.csv', header=None)
dt_rtr = pd.read_csv('telapsed_rtr.csv', header=None)
dt_cg = pd.read_csv('telapsed_cg.csv', header=None)
dt_cg2 = pd.read_csv('telapsed_cg2.csv', header=None)
dt_admm = pd.read_csv('telapsed_admm.csv', header=None)
dt_cgal1 = pd.read_csv('telapsed_cgal1.csv', header=None)
dt_cgal2 = pd.read_csv('telapsed_cgal2.csv', header=None)
dt_lorads = pd.read_csv('telapsed_lorads.csv', header=None)

# dt_bcm[dt_bcm > 60] = 60
# dt_rgd[dt_rgd > 60] = 60
# dt_rtr[dt_rtr > 60] = 60
# dt_cg[dt_cg > 60] = 60
# dt_cg2[dt_cg2 > 60] = 60
# dt_admm[dt_admm > 60] = 60
# dt_cgal1[dt_cgal1 > 60] = 60
# dt_cgal2[dt_cgal2 > 60] = 60
# dt_lorads[dt_lorads > 60] = 60

t_bcm = dt_bcm.apply(lambda row: row.dropna().tolist(), axis=1).tolist()
t_bcm2 = dt_bcm2.apply(lambda row: row.dropna().tolist(), axis=1).tolist()
t_rgd = dt_rgd.apply(lambda row: row.dropna().tolist(), axis=1).tolist()
t_rtr = dt_rtr.apply(lambda row: row.dropna().tolist(), axis=1).tolist()
t_cg = dt_cg.apply(lambda row: row.dropna().tolist(), axis=1).tolist()
t_cg2 = dt_cg2.apply(lambda row: row.dropna().tolist(), axis=1).tolist()
t_admm = dt_admm.apply(lambda row: row.dropna().tolist(), axis=1).tolist()
t_cgal1 = dt_cgal1.apply(lambda row: row.dropna().tolist(), axis=1).tolist()
t_cgal2 = dt_cgal2.apply(lambda row: row.dropna().tolist(), axis=1).tolist()
t_lorads = dt_lorads.apply(lambda row: row.dropna().tolist(), axis=1).tolist()

#%%

interpolation = "linear"


interpolated_curve_bcm = np.array([
    interp1d(t, f, kind=interpolation, fill_value="extrapolate")(common_time)
    for t, f in zip(t_bcm, f_bcm)
])

interpolated_curve_bcm2 = np.array([
    interp1d(t, f, kind=interpolation, fill_value="extrapolate")(common_time)
    for t, f in zip(t_bcm2, f_bcm2)
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

interpolated_curve_admm = np.array([
    interp1d(t, f, kind=interpolation, fill_value="extrapolate")(common_time)
    for t, f in zip(t_admm, f_admm)
])

interpolated_curve_cgal1 = np.array([
    interp1d(t, f, kind=interpolation, fill_value="extrapolate")(common_time)
    for t, f in zip(t_cgal1, f_cgal1)
])

interpolated_curve_cgal2 = np.array([
    interp1d(t, f, kind=interpolation, fill_value="extrapolate")(common_time)
    for t, f in zip(t_cgal2, f_cgal2)
])

interpolated_curve_lorads = np.array([
    interp1d(t, f, kind=interpolation, fill_value="extrapolate")(common_time)
    for t, f in zip(t_lorads, f_lorads)
])


mean_curve_bcm = -np.mean(interpolated_curve_bcm, axis=0)
mean_curve_bcm2 = -np.mean(interpolated_curve_bcm2, axis=0)
mean_curve_rgd = -np.mean(interpolated_curve_rgd, axis=0)
mean_curve_rtr = -np.mean(interpolated_curve_rtr, axis=0)
mean_curve_cg = -np.mean(interpolated_curve_cg, axis=0)
mean_curve_cg2 = -np.mean(interpolated_curve_cg2, axis=0)
mean_curve_admm = -np.mean(interpolated_curve_admm, axis=0)
mean_curve_cgal1 = -np.mean(interpolated_curve_cgal1, axis=0)
mean_curve_cgal2 = -np.mean(interpolated_curve_cgal2, axis=0)
mean_curve_lorads = -np.mean(interpolated_curve_lorads, axis=0)

fig0 = plt.figure(0)


line_bcm,=plt.plot(common_time, mean_curve_bcm,color = "orange", linestyle = "solid", linewidth=2, label="BCM")
line_bcm2,=plt.plot(common_time, mean_curve_bcm2,color = "black", linestyle = "dotted", linewidth=2, label="BCM2")
line_rgd,=plt.plot(common_time, mean_curve_rgd,color='gold',   linewidth=2,linestyle = "solid", label="RGD")
line_rtr,=plt.plot(common_time, mean_curve_rtr,color='blue',     linestyle = "dashed", linewidth=2, label="RTR")
line_gfw,=plt.plot(common_time, mean_curve_cg,color = "lime", linestyle = "solid", linewidth=2, label="GFW")
# plt.plot(common_time, mean_curve_cg2,color = "purple", linestyle = "densely dotted", linewidth=2, label="CG + RTR")
line_gfw_rtr,=plt.plot(common_time, mean_curve_cg2,color = "red", linewidth=2, label="GFW & RTR", linestyle = "dotted")
line_admm,=plt.plot(common_time, mean_curve_admm,color = "blueviolet", linewidth=2, label="ADMM", linestyle = "solid")
line_cgal1,=plt.plot(common_time, mean_curve_cgal1,color = "aqua", linewidth=2, label="SCGAL R=10 ", linestyle = "solid")
line_cgal2,=plt.plot(common_time, mean_curve_cgal2,color = "magenta", linewidth=2, label="SCGAL R=200", linestyle = "dotted")
line_lorads,=plt.plot(common_time, mean_curve_lorads,color = "green", linewidth=2, label="LoRADS*", linestyle = "solid")


# plt.xscale("log")

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


plt.xlim((6,8))


ci = 1
plt.xlim((55,60))
plt.ylim((mean_curve_rtr[-1]-ci,mean_curve_rtr[-1]+ci))

plt.tight_layout()  # Automatically adjusts all margins
plt.legend(handles=[line_gfw_rtr,line_gfw,line_rtr,line_rgd,line_bcm2]) 

# # fig0.savefig("cg_rtr_comparison_T_"+str(num_curves)+"_last_second.png",format ="png")
fig0.savefig("cg_rtr_comparison_T_"+str(num_curves)+"_last_second.eps",format ="eps")
fig0.savefig("cg_rtr_comparison_T_"+str(num_curves)+"_last_second.pdf",format ="pdf")


# fig0.savefig("cg_rtr_comparison_T_"+str(num_curves)+"_last_second.eps",format ="eps",bbox_inches='tight', pad_inches=0.05)
# fig0.savefig("cg_rtr_comparison_T_"+str(num_curves)+"_last_second.pdf",format ="pdf",bbox_inches='tight', pad_inches=0.05)
import os
import pandas as pd
import numpy as np
import pathnavigator
from tqdm import tqdm
import clt
root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\SD6InternalVariability"
root_dir = rf"/Users/{pathnavigator.user}/Documents/GitHub/SD6InternalVariability"
pn = pathnavigator.create(root_dir)
pn.code.chdir()
from anova_utils import (
    get_sum_sq_over_years,
    get_mu_sd_dfs_over_seeds,
    )
# Collected simulated results
df_sys_all = pd.read_parquet(pn.outputs.ANOVA.get("df_sys_all.parquet"))
vlist = ['ST', 'CF', 'Wi', 'CSC']

#%% Compute ANONA over seeds
mu_dict_all = {}
sd_dict_all = {}
sum_sq_nor_seed67 = {} 
sum_sq_nor_seed1 = {} 
sum_sq_nor_seed2 = {} 
levene_all = {} 
Omnibus_all = {}
l = len(vlist)
nRe = 15 # Number of boostrapping time series resamples
nCd = 5 # Number of corn maps

# Use all available data (main results in the paper) => sum_sq_nor
if os.path.exists(pn.outputs.ANOVA.get()/f"anova_mu_sum_sq_nRe{nRe}_nCd{nCd}.h5") is False: 
    for i, v in tqdm(enumerate(vlist)):
        mu_dict_all[v], sd_dict_all[v], levene_all[v], Omnibus_all[v], sum_sq_nor = get_mu_sd_dfs_over_seeds(v, df_sys_all, to_fraction=False, typ=1)
        sum_sq_nor_seed67[v] = sum_sq_nor[0]
        sum_sq_nor_seed1[v] = sum_sq_nor[1]
        sum_sq_nor_seed2[v] = sum_sq_nor[2]
        print(i+1,"/", l)
    
    clt.io.to_pd_hdf5(data=mu_dict_all, file_path=pn.outputs.ANOVA.get()/f"anova_mu_sum_sq_nRe{nRe}_nCd{nCd}.h5")
    clt.io.to_pd_hdf5(data=sd_dict_all, file_path=pn.outputs.ANOVA.get()/f"anova_sd_sum_sq_nRe{nRe}_nCd{nCd}.h5")
    clt.io.to_pd_hdf5(data=sum_sq_nor_seed67, file_path=pn.outputs.ANOVA.get()/f"anova_sum_sq_nor_seed67_nRe{nRe}_nCd{nCd}.h5")
    clt.io.to_pd_hdf5(data=sum_sq_nor_seed1, file_path=pn.outputs.ANOVA.get()/f"anova_sum_sq_nor_seed1_nRe{nRe}_nCd{nCd}.h5")
    clt.io.to_pd_hdf5(data=sum_sq_nor_seed2, file_path=pn.outputs.ANOVA.get()/f"anova_sum_sq_nor_seed2_nRe{nRe}_nCd{nCd}.h5")
    clt.io.to_pd_hdf5(data=mu_dict_all, file_path=pn.outputs.ANOVA.get()/f"anova_levene_nRe{nRe}_nCd{nCd}.h5")
    clt.io.to_pd_hdf5(data=sd_dict_all, file_path=pn.outputs.ANOVA.get()/f"anova_Omnibus_nRe{nRe}_nCd{nCd}.h5")

# We have proved there is no error term. Therefore, we can calculate fraction with IV included directly. (== to_fraction=True)
mu_dict_all = clt.io.read_pd_hdf5(pn.outputs.ANOVA.get()/"anova_mu_sum_sq_nRe15_nCd5.h5")
sd_dict_all = clt.io.read_pd_hdf5(pn.outputs.ANOVA.get()/"anova_sd_sum_sq_nRe15_nCd5.h5")
mu_dict = {k: v[["C(Pr)", "C(Cr)", "C(Co)", "Interaction terms"]] for k, v in mu_dict_all.items()}
sd_dict = {k: v[["C(Pr)", "C(Cr)", "C(Co)", "Interaction terms"]] for k, v in sd_dict_all.items()}

# No error term
for v in vlist:
    mu_dict[v]["IV"] = mu_dict_all[v]["Residual"] 
    mu_dict[v]["Error"] = 0 

clt.io.to_pd_hdf5(data=mu_dict, file_path=pn.outputs.ANOVA.get()/"anova_mu_sum_sq_withIV_seperated.h5")

for v in vlist:
    mu_dict[v] = mu_dict[v].div(mu_dict[v].sum(axis=1), axis=0)
clt.io.to_pd_hdf5(data=mu_dict, file_path=pn.outputs.ANOVA.get()/"anova_mu_fraction_withIV_seperated.h5")

#%% Testing convergence with different nRe and nCd
for nRe in [10,12,14]:
    df_sys_all_ = df_sys_all[df_sys_all["Re"]<nRe]
    mu_dict_all = {}
    sd_dict_all = {}
    sum_sq_nor_seed67 = {} 
    sum_sq_nor_seed1 = {} 
    sum_sq_nor_seed2 = {} 
    levene_all = {} 
    Omnibus_all = {}
    l = len(vlist)
    #nRe = 15
    nCd = 5
    
    if os.path.exists(pn.outputs.ANOVA.get()/f"anova_mu_sum_sq_nRe{nRe}_nCd{nCd}.h5") is False: 
        for i, v in tqdm(enumerate(vlist)):
            mu_dict_all[v], sd_dict_all[v], levene_all[v], Omnibus_all[v], sum_sq_nor = get_mu_sd_dfs_over_seeds(
                v, df_sys_all_, to_fraction=False, typ=1)
            sum_sq_nor_seed67[v] = sum_sq_nor[0]
            sum_sq_nor_seed1[v] = sum_sq_nor[1]
            sum_sq_nor_seed2[v] = sum_sq_nor[2]
            print(i+1,"/", l)
        
        clt.io.to_pd_hdf5(data=mu_dict_all, file_path=pn.outputs.ANOVA.get()/f"anova_mu_sum_sq_nRe{nRe}_nCd{nCd}.h5")
        clt.io.to_pd_hdf5(data=sd_dict_all, file_path=pn.outputs.ANOVA.get()/f"anova_sd_sum_sq_nRe{nRe}_nCd{nCd}.h5")
        clt.io.to_pd_hdf5(data=sum_sq_nor_seed67, file_path=pn.outputs.ANOVA.get()/f"anova_sum_sq_nor_seed67_nRe{nRe}_nCd{nCd}.h5")
        clt.io.to_pd_hdf5(data=sum_sq_nor_seed1, file_path=pn.outputs.ANOVA.get()/f"anova_sum_sq_nor_seed1_nRe{nRe}_nCd{nCd}.h5")
        clt.io.to_pd_hdf5(data=sum_sq_nor_seed2, file_path=pn.outputs.ANOVA.get()/f"anova_sum_sq_nor_seed2_nRe{nRe}_nCd{nCd}.h5")
        clt.io.to_pd_hdf5(data=mu_dict_all, file_path=pn.outputs.ANOVA.get()/f"anova_levene_nRe{nRe}_nCd{nCd}.h5")
        clt.io.to_pd_hdf5(data=sd_dict_all, file_path=pn.outputs.ANOVA.get()/f"anova_Omnibus_nRe{nRe}_nCd{nCd}.h5")

nRe_mu_dict = {}
for nRe in [10,12,14,15]:
    # We have proved there is no error term. Therefore, we can calculate fraction with IV included directly. (== to_fraction=True)
    mu_dict_all = clt.io.read_pd_hdf5(pn.outputs.ANOVA.get()/f"anova_mu_sum_sq_nRe{nRe}_nCd5.h5")
    mu_dict = {k: v[["C(Pr)", "C(Cr)", "C(Co)", "Interaction terms"]] for k, v in mu_dict_all.items()}
    # No error term
    for v in vlist:
        mu_dict[v]["IV"] = mu_dict_all[v]["Residual"] 
        mu_dict[v]["Error"] = 0 
    nRe_mu_dict[nRe] = mu_dict
    
nRe_iv_dict = {}
for v in vlist:
    df = pd.DataFrame()
    for nRe in [10,12,14,15]:
        df[nRe] = nRe_mu_dict[nRe][v]["IV"]/nRe_mu_dict[nRe][v].sum(axis=1)
    nRe_iv_dict[v] = df
    df.mean().plot(ylim=[0,1])
    print(df.mean())
clt.io.to_pd_hdf5(data=nRe_iv_dict, file_path=pn.outputs.ANOVA.get()/"anova_iv_fraction_nRe.h5")

#%%
for nCd in [3,4]:
    df_sys_all_ = df_sys_all[df_sys_all["Cd"]<nCd]
    mu_dict_all = {}
    sd_dict_all = {}
    sum_sq_nor_seed67 = {} 
    sum_sq_nor_seed1 = {} 
    sum_sq_nor_seed2 = {} 
    levene_all = {} 
    Omnibus_all = {}
    l = len(vlist)
    nRe = 15
    #nCd = 5
    if os.path.exists(pn.outputs.ANOVA.get()/f"anova_mu_sum_sq_nRe{nRe}_nCd{nCd}.h5") is False: 
        for i, v in tqdm(enumerate(vlist)):
            mu_dict_all[v], sd_dict_all[v], levene_all[v], Omnibus_all[v], sum_sq_nor = get_mu_sd_dfs_over_seeds(
                v, df_sys_all_, to_fraction=False, typ=1)
            sum_sq_nor_seed67[v] = sum_sq_nor[0]
            sum_sq_nor_seed1[v] = sum_sq_nor[1]
            sum_sq_nor_seed2[v] = sum_sq_nor[2]
            print(i+1,"/", l)
        
        nRe = 15
        #nCd = 5
        
        clt.io.to_pd_hdf5(data=mu_dict_all, file_path=pn.outputs.ANOVA.get()/f"anova_mu_sum_sq_nRe{nRe}_nCd{nCd}.h5")
        clt.io.to_pd_hdf5(data=sd_dict_all, file_path=pn.outputs.ANOVA.get()/f"anova_sd_sum_sq_nRe{nRe}_nCd{nCd}.h5")
        clt.io.to_pd_hdf5(data=sum_sq_nor_seed67, file_path=pn.outputs.ANOVA.get()/f"anova_sum_sq_nor_seed67_nRe{nRe}_nCd{nCd}.h5")
        clt.io.to_pd_hdf5(data=sum_sq_nor_seed1, file_path=pn.outputs.ANOVA.get()/f"anova_sum_sq_nor_seed1_nRe{nRe}_nCd{nCd}.h5")
        clt.io.to_pd_hdf5(data=sum_sq_nor_seed2, file_path=pn.outputs.ANOVA.get()/f"anova_sum_sq_nor_seed2_nRe{nRe}_nCd{nCd}.h5")
        clt.io.to_pd_hdf5(data=mu_dict_all, file_path=pn.outputs.ANOVA.get()/f"anova_levene_nRe{nRe}_nCd{nCd}.h5")
        clt.io.to_pd_hdf5(data=sd_dict_all, file_path=pn.outputs.ANOVA.get()/f"anova_Omnibus_nRe{nRe}_nCd{nCd}.h5")

nCd_mu_dict = {}
for nCd in [3,4,5]:
    # We have proved there is no error term. Therefore, we can calculate fraction with IV included directly. (== to_fraction=True)
    mu_dict_all = clt.io.read_pd_hdf5(pn.outputs.ANOVA.get()/f"anova_mu_sum_sq_nRe15_nCd{nCd}.h5")
    mu_dict = {k: v[["C(Pr)", "C(Cr)", "C(Co)", "Interaction terms"]] for k, v in mu_dict_all.items()}
    # No error term
    for v in vlist:
        mu_dict[v]["IV"] = mu_dict_all[v]["Residual"] 
        mu_dict[v]["Error"] = 0 
    nCd_mu_dict[nCd] = mu_dict
    
nCd_iv_dict = {}
for v in vlist:
    df = pd.DataFrame()
    for nCd in [3,4,5]:
        df[nCd] = nCd_mu_dict[nCd][v]["IV"]/nCd_mu_dict[nCd][v].sum(axis=1)
    nCd_iv_dict[v] = df
    df.mean().plot(ylim=[0,1])
    print(df.mean())
clt.io.to_pd_hdf5(data=nCd_iv_dict, file_path=pn.outputs.ANOVA.get()/"anova_iv_fraction_nCd.h5")

# import matplotlib.pyplot as plt

# fig, axes = plt.subplots(2, 2, figsize=(6, 4), sharex=True)
# axes = axes.flatten()

# for ax, v in zip(axes, vlist):
#     df = nRe_iv_dict[v]
    
#     for nRe in df.columns:
#         ax.plot(df.index, df[nRe], label=f"nRe={nRe}")
    
#     ax.set_title(f"v = {v}")
#     ax.set_xlabel("Index")
#     ax.set_ylabel("IV")
#     ax.legend()

# plt.tight_layout()
# plt.show()

#%%
seeds_mu_dict = {}
for seed in [1,2,67]:
    # We have proved there is no error term. Therefore, we can calculate fraction with IV included directly. (== to_fraction=True)
    mu_dict_all = clt.io.read_pd_hdf5(pn.outputs.ANOVA.get()/f"anova_sum_sq_nor_seed{seed}_nRe15_nCd5.h5")
    mu_dict = {k: v[["C(Pr)", "C(Cr)", "C(Co)", "Interaction terms"]] for k, v in mu_dict_all.items()}
    # No error term
    for v in vlist:
        mu_dict[v]["IV"] = mu_dict_all[v]["Residual"] 
        mu_dict[v]["Error"] = 0 
    seeds_mu_dict[seed] = mu_dict
    
seeds_iv_dict = {}
for v in vlist:
    df = pd.DataFrame()
    for seed in [1,2,67]:
        df[seed] = seeds_mu_dict[seed][v]["IV"]/seeds_mu_dict[seed][v].sum(axis=1)
    seeds_iv_dict[v] = df
    df.mean().plot(ylim=[0,1])
    print(df.mean())
clt.io.to_pd_hdf5(data=seeds_iv_dict, file_path=pn.outputs.ANOVA.get()/"anova_iv_fraction_seeds.h5")


########################## Irrigation norm analysis ###########################
#%% Assign regimes and save anova results
# Define thresholds (Obv avg over 2013-2022)
wi_thresh = 22.76
st_thresh = 17.97
merge_keys = ['Pr', 'Cr', 'Co', 'Cd', 'Re', 'Seed']

# Prepare df_sys_all_mean with regime labels
df_sys_all_mean = (
    df_sys_all[df_sys_all["Year"] != 2012]
    .drop(columns="Year")
    .groupby(merge_keys, as_index=False)
    .mean()
)

# Assign regimes using np.select
df_sys_all_mean["Wi_regime"] = np.where(df_sys_all_mean["Wi"] <= wi_thresh, "lower", "higher")
df_sys_all_mean["ST_regime"] = np.where(df_sys_all_mean["ST"] <= st_thresh, "lower", "higher")

# Merge regimes back into original df_sys_all
df_sys_all = df_sys_all.merge(
    df_sys_all_mean[merge_keys + ["Wi_regime", "ST_regime"]],
    on=merge_keys,
    how='left'
)

df_Wi_regime = df_sys_all_mean.groupby("Wi_regime").mean(numeric_only=True)
df_ST_regime = df_sys_all_mean.groupby("ST_regime").mean(numeric_only=True)


max_vals = pd.concat([df_Wi_regime, df_ST_regime]).max()
df_Wi_regime_norm = df_Wi_regime / max_vals
df_ST_regime_norm = df_ST_regime / max_vals

df_Wi_regime_norm.to_csv(pn.figures.data_for_plotting.get()/"df_Wi_regime_norm.csv")
df_ST_regime_norm.to_csv(pn.figures.data_for_plotting.get()/"df_ST_regime_norm.csv")

#%% Regime analysis
r"""
ax = df_sys_all_mean['Wi'].hist(bins=100) # Obv 22.76
ax.axvline(22.76, c="r")
ax.set_xlabel("Wi")
plt.show()

ax = df_sys_all_mean['ST'].hist(bins=100) # Obv 22.76
ax.axvline(17.97, c="r")
ax.set_xlabel("ST")
plt.show()
"""

vlist = ['ST', 'Wi', 'RF', 'CF', 'OF', 'CSC', 'TP']
# Fraction
for v_regime in ["Wi_regime"]:
    for regime in ['higher', 'lower']:
        df = df_sys_all[df_sys_all[v_regime]==regime]
        mu_dict = {}
        sd_dict = {}
        for v in vlist:
            #mu_dict[v], sd_dict[v] = get_mu_sd_dfs_over_seeds(v, df, to_fraction=True)
            mu_dict[v], sd_dict[v], _, _, _ = get_mu_sd_dfs_over_seeds(v, df, to_fraction=True, typ=1)

        clt.io.to_pd_hdf5(data=mu_dict, file_path=pn.outputs.ANOVA.get()/f"anova_mu_fraction_{v_regime}_{regime}.h5")
        clt.io.to_pd_hdf5(data=sd_dict, file_path=pn.outputs.ANOVA.get()/f"anova_sd_fraction_{v_regime}_{regime}.h5")
# Sum of square
# for v_regime in ["Wi_regime"]:
#     for regime in ['higher', 'lower']:
#         df = df_sys_all[df_sys_all[v_regime]==regime]
#         mu_dict = {}
#         sd_dict = {}
#         for v in vlist:
#             mu_dict[v], sd_dict[v] = get_mu_sd_dfs_over_seeds(v, df, to_fraction=False)

#         clt.io.to_pd_hdf5(data=mu_dict, file_path=pn.outputs.ANOVA.get()/f"anova_mu_sum_sq_{v_regime}_{regime}.h5")
#         clt.io.to_pd_hdf5(data=sd_dict, file_path=pn.outputs.ANOVA.get()/f"anova_sd_sum_sq_{v_regime}_{regime}.h5")






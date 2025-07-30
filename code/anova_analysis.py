import pandas as pd
import numpy as np
import pathnavigator
import clt
root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\SD6InternalVariability"
pn = pathnavigator.create(root_dir)
pn.code.chdir()
from anova_utils import (
    get_sum_sq_over_years,
    get_mu_sd_dfs_over_seeds,
    plot_anova_sum_sq_fraction,
    plot_anova_sum_sq,
    plot_regime_comparison
    )
# Collected simulated results
df_sys_all = pd.read_csv(pn.outputs.ANOVA.get("df_sys_all.csv"))
vlist = ['ST', 'CF', 'Wi', 'CSC']
#%% Compute ANONA over seeds
mu_dict_all = {}
sd_dict_all = {}
for v in vlist:
    mu_dict_all[v], sd_dict_all[v] = get_mu_sd_dfs_over_seeds(v, df_sys_all, to_fraction=False)
clt.io.to_pd_hdf5(data=mu_dict_all, file_path=pn.outputs.ANOVA.get()/"anova_mu_sum_sq.h5")
clt.io.to_pd_hdf5(data=sd_dict_all, file_path=pn.outputs.ANOVA.get()/"anova_sd_sum_sq.h5")

#%% Conduct ANOVA decomposition with fixed IV for estimating ANOVA model structural error (No need to rerun)
cd_vals = df_sys_all["Cd"].unique()
re_vals = df_sys_all["Re"].unique()
df_sys_all_noIV = df_sys_all[(df_sys_all["Cd"] == 0) & (df_sys_all["Re"] == 0)]
pn.outputs.ANOVA.mkdir("fixed_IV")
# Normalized
for cd in cd_vals:
    for re in re_vals:
        mu_dict = {}
        sd_dict = {}
        for v in vlist:
            mu_dict[v], sd_dict[v] = get_mu_sd_dfs_over_seeds(v, df_sys_all_noIV, to_fraction=False)
        clt.io.to_pd_hdf5(data=mu_dict, file_path=pn.outputs.ANOVA.fixed_IV.get()/f"anova_mu_sum_sq_Cd{cd}_Re{re}.h5")
        clt.io.to_pd_hdf5(data=sd_dict, file_path=pn.outputs.ANOVA.fixed_IV.get()/f"anova_sd_sum_sq_Cd{cd}_Re{re}.h5")

#%% Compute average residual with fixed IV => estimated ANOVA model structural error
from collections import defaultdict

mu_accumulator = defaultdict(list)
sd_accumulator = defaultdict(list)
for cd in cd_vals:
    for re in re_vals:
        # Load mu and sd HDF5 files
        mu_path = pn.outputs.ANOVA.fixed_IV.get() / f"anova_mu_sum_sq_Cd{cd}_Re{re}.h5"
        sd_path = pn.outputs.ANOVA.fixed_IV.get() / f"anova_sd_sum_sq_Cd{cd}_Re{re}.h5"

        mu_dict = clt.io.read_pd_hdf5(mu_path)
        sd_dict = clt.io.read_pd_hdf5(sd_path)

        # Compute column-wise averages
        for v in vlist:
            mu_accumulator[v].append(mu_dict[v])
            sd_accumulator[v].append(sd_dict[v])

# Compute the average over all Cd-Re combinations for each variable
mu_avg_all = {v: pd.concat(mu_accumulator[v]).groupby(level=0).mean() for v in vlist}
sd_avg_all = {v: pd.concat(sd_accumulator[v]).groupby(level=0).mean() for v in vlist}
clt.io.to_pd_hdf5(data=mu_avg_all, file_path=pn.outputs.ANOVA.get()/"anova_mu_sum_sq_avg_fixedIV.h5")
clt.io.to_pd_hdf5(data=sd_avg_all, file_path=pn.outputs.ANOVA.get()/"anova_sd_sum_sq_avg_fixedIV.h5")


#%% Calculate IV
r"""
Estimated Internal Variability≈Var(residual,all) − Var(residual,fixed)
​
This assumes:

- Internal variability and model error are additive and uncorrelated (a key assumption).
- The simulation noise or numerical error is stable across runs (not biased by fixed seed).

You can then reinterpret your original residual term in the main ANOVA as:
Residual = Model Error + Internal Variability
"""
mu_dict_fixedIV = clt.io.read_pd_hdf5(pn.outputs.ANOVA.get()/"anova_mu_sum_sq_avg_fixedIV.h5")
sd_dict_fixedIV = clt.io.read_pd_hdf5(pn.outputs.ANOVA.get()/"anova_sd_sum_sq_avg_fixedIV.h5")

mu_dict_all = clt.io.read_pd_hdf5(pn.outputs.ANOVA.get()/"anova_mu_sum_sq.h5")
sd_dict_all = clt.io.read_pd_hdf5(pn.outputs.ANOVA.get()/"anova_sd_sum_sq.h5")

mu_dict = {k: v[["Pr", "Cr", "Co", "Interaction terms"]] for k, v in mu_dict_all.items()}
sd_dict = {k: v[["Pr", "Cr", "Co", "Interaction terms"]] for k, v in sd_dict_all.items()}

for v in vlist:
    mu_dict[v]["IV"] = mu_dict_all[v]["Residual"] - mu_dict_fixedIV[v]["Residual"]
    mu_dict[v]["Error"] = mu_dict_fixedIV[v]["Residual"]

clt.io.to_pd_hdf5(data=mu_dict, file_path=pn.outputs.ANOVA.get()/"anova_mu_sum_sq_withIV_seperated.h5")
#clt.io.to_pd_hdf5(data=sd_dict, file_path=pn.outputs.exp4_ANOVA.get()/"anova_sd_sum_sq_withIV_seperated.h5")

for v in vlist:
    mu_dict[v] = mu_dict[v].div(mu_dict[v].sum(axis=1), axis=0)
clt.io.to_pd_hdf5(data=mu_dict, file_path=pn.outputs.ANOVA.get()/"anova_mu_fraction_withIV_seperated.h5")


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
            mu_dict[v], sd_dict[v] = get_mu_sd_dfs_over_seeds(v, df, to_fraction=True)

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






import pathnavigator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker as mticker
import clt
root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\SD6InternalVariability"
root_dir = rf"/Users/{pathnavigator.user}/Documents/GitHub/SD6InternalVariability"
pn = pathnavigator.create(root_dir)
pn.code.chdir()

nRe_iv_dict = clt.io.read_pd_hdf5(pn.outputs.ANOVA.get()/"anova_iv_fraction_nRe.h5")
nCd_iv_dict = clt.io.read_pd_hdf5(pn.outputs.ANOVA.get()/"anova_iv_fraction_nCd.h5")
seeds_iv_dict = clt.io.read_pd_hdf5(pn.outputs.ANOVA.get()/"anova_iv_fraction_seeds.h5")
vlist = ['ST', 'CF', 'Wi', 'CSC']
var_dict = {
    'ST': 'Saturated\nthickness',
    'Wi': 'Withdrawal',
    'RF': 'Rainfed\npercent cover',
    'CF': 'Corn percent\ncover',
    'OF': 'Other crops\npercent cover',
    'CSC': 'Behavioral state\nchange percentage',
    'TP': 'Total\nprofit'
    }
#%%


#fig, ax = plt.subplots(figsize=(6, 4))

fig, axes = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
axes = axes.flatten()
ax = axes[0]
# ---- build summary table: rows=variable key, cols=seed, values=column mean ----
summary = pd.DataFrame(
    {k: df.mean(axis=0) for k, df in nRe_iv_dict.items()}
).T  # transpose so index = keys, columns = seeds

# ensure consistent column order (numeric seeds)
summary.columns = pd.to_numeric(summary.columns)
summary = summary.reindex(sorted(summary.columns), axis=1)

# ---- grouped bar chart ----
x_labels = summary.index.tolist()
seed_cols = summary.columns.tolist()

x = np.arange(len([var_dict.get(k, k) for k in x_labels]))
width = 0.8 / len(seed_cols)
for i, seed in enumerate(seed_cols):
    ax.bar(x + i * width, summary[seed].to_numpy(), width, label=str(seed))

ax.set_xticks(x + (len(seed_cols) - 1) * width / 2)
ax.set_xticklabels(x_labels)
#ax.set_ylabel("Mean fraction of variance\nexplained by internal variability")
ax.legend(
    title="Number of\nbootstrapped\nyear sequences", 
    frameon=False,
    loc="center right",
    alignment="left",
    bbox_to_anchor=(1.3, 0.5),
    )


ax = axes[1]
# ---- build summary table: rows=variable key, cols=seed, values=column mean ----
summary = pd.DataFrame(
    {k: df.mean(axis=0) for k, df in nCd_iv_dict.items()}
).T  # transpose so index = keys, columns = seeds

# ensure consistent column order (numeric seeds)
summary.columns = pd.to_numeric(summary.columns)
summary = summary.reindex(sorted(summary.columns), axis=1)

# ---- grouped bar chart ----
x_labels = summary.index.tolist()
seed_cols = summary.columns.tolist()

x = np.arange(len(x_labels))
width = 0.8 / len(seed_cols)
for i, seed in enumerate(seed_cols):
    ax.bar(x + i * width, summary[seed].to_numpy(), width, label=str(seed))

ax.set_xticks(x + (len(seed_cols) - 1) * width / 2)
ax.set_xticklabels([var_dict.get(k, k) for k in x_labels])
ax.set_ylabel("Mean fraction of variance\nexplained by internal variability")
ax.legend(
    title="Number of\nresampled\ninitial crop maps", 
    frameon=False,
    loc="center right",
    alignment="left",
    bbox_to_anchor=(1.31, 0.5),
    )

ax = axes[2]
# ---- build summary table: rows=variable key, cols=seed, values=column mean ----
summary = pd.DataFrame(
    {k: df.mean(axis=0) for k, df in seeds_iv_dict.items()}
).T  # transpose so index = keys, columns = seeds

# ensure consistent column order (numeric seeds)
summary.columns = pd.to_numeric(summary.columns)
summary = summary.reindex(sorted(summary.columns), axis=1)

# ---- grouped bar chart ----
x_labels = summary.index.tolist()
seed_cols = summary.columns.tolist()

x = np.arange(len(x_labels))
width = 0.8 / len(seed_cols)
for i, seed in enumerate(seed_cols):
    ax.bar(x + i * width, summary[seed].to_numpy(), width, label=str(seed))

ax.set_xticks(x + (len(seed_cols) - 1) * width / 2)
ax.set_xticklabels([var_dict.get(k, k) for k in x_labels])
#ax.set_ylabel("Mean fraction of variance\nexplained by internal variability")
ax.set_xlabel("Output variable")
ax.set_ylim([0,0.8])
ax.legend(
    title="Seed", 
    frameon=False,
    loc="center right",
    alignment="left",
    bbox_to_anchor=(1.175, 0.5),
    )

panel_labels = ["(a)", "(b)", "(c)"]

for ax, label in zip(axes, panel_labels):
    ax.text(
        0.01, 0.98, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=12,
        fontweight="bold"
    )

plt.tight_layout()
plt.savefig(pn.figures.get() / "figS6_convergence_avg_all.jpg", dpi=300)
plt.show()


#%% nRe timeseries bar plot
fig, axes = plt.subplots(2, 2, figsize=(6, 5), sharex=True)
axes = axes.flatten()

handles, labels = None, None

for ax, v in zip(axes, vlist):
    if v in ["CSC", "CF"]:
        df = nRe_iv_dict[v].copy()*100
    else:
        df = nRe_iv_dict[v].copy()

    # keep only 2013–2022 and ensure year index is integer
    df.index = df.index.astype(int)
    df = df.loc[(df.index >= 2013) & (df.index <= 2022)]

    years = df.index.to_numpy()
    x = np.arange(len(years))               # categorical positions
    n_groups = df.shape[1]                  # number of nRe columns
    width = 0.8 / n_groups                  # total group width = 0.8

    for i, nRe in enumerate(df.columns):
        bars = ax.bar(x + i * width, df[nRe].to_numpy(), width, label=f"nRe={nRe}")

    # capture legend once
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()

    # x ticks as years (integers) and rotated
    ax.set_xticks(x + (n_groups - 1) * width / 2)
    ax.set_xticklabels(years, rotation=90)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=False))

    if v in ["Wi", "CSC"]:
        ax.set_xlabel("Year")
    if v in ["Wi", "ST"]:
        ax.set_ylabel("Fraction of variance\nexplained by\ninternal variability")
    ax.set_title(var_dict[v])

# figure-level legend (outside right)
fig.legend(
    handles, [l[-2:] for l in labels],
    title="Number of\nbootstrapped\nyear sequences",
    loc="center right",
    frameon=False,
    alignment="left",
    bbox_to_anchor=(1.2, 0.5),
)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(pn.figures.get() / "figS6_convergence_nRe.jpg", dpi=300)
plt.show()

#%% nCd timeseries bar plot
fig, axes = plt.subplots(2, 2, figsize=(6, 5), sharex=True)
axes = axes.flatten()

handles, labels = None, None

for ax, v in zip(axes, vlist):
    if v in ["CSC", "CF"]:
        df = nCd_iv_dict[v].copy()*100
    else:
        df = nCd_iv_dict[v].copy()
    # keep only 2013–2022 and ensure year index is integer
    df.index = df.index.astype(int)
    df = df.loc[(df.index >= 2013) & (df.index <= 2022)]

    years = df.index.to_numpy()
    x = np.arange(len(years))               # categorical positions
    n_groups = df.shape[1]                  # number of nCd columns
    width = 0.8 / n_groups                  # total group width = 0.8

    for i, nCd in enumerate(df.columns):
        bars = ax.bar(x + i * width, df[nCd].to_numpy(), width, label=f"nCd={nCd}")

    # capture legend once
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()

    # x ticks as years (integers) and rotated
    ax.set_xticks(x + (n_groups - 1) * width / 2)
    ax.set_xticklabels(years, rotation=90)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=False))

    if v in ["Wi", "CSC"]:
        ax.set_xlabel("Year")
    if v in ["Wi", "ST"]:
        ax.set_ylabel("Fraction of variance\nexplained by\ninternal variability")
    ax.set_title(var_dict[v])

# figure-level legend (outside right)
fig.legend(
    handles, [l[-1:] for l in labels],
    title="Number of\nresampled\ninitial crop maps",
    loc="center right",
    frameon=False,
    alignment="left",
    bbox_to_anchor=(1.2, 0.5),
)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(pn.figures.get() / "figS6_convergence_nCd.jpg", dpi=300)
plt.show()

#%% seed timeseries bar plot
fig, axes = plt.subplots(2, 2, figsize=(6, 5), sharex=True)
axes = axes.flatten()

handles, labels = None, None

for ax, v in zip(axes, vlist):
    if v in ["CSC", "CF"]:
        df = seeds_iv_dict[v].copy()*100
    else:
        df = seeds_iv_dict[v].copy()

    # keep only 2013–2022 and ensure year index is integer
    df.index = df.index.astype(int)
    df = df.loc[(df.index >= 2013) & (df.index <= 2022)]

    years = df.index.to_numpy()
    x = np.arange(len(years))               # categorical positions
    n_groups = df.shape[1]                  # number of nRe columns
    width = 0.8 / n_groups                  # total group width = 0.8

    for i, seed in enumerate(df.columns):
        bars = ax.bar(x + i * width, df[seed].to_numpy(), width, label=f"{seed}")

    # capture legend once
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()

    # x ticks as years (integers) and rotated
    ax.set_xticks(x + (n_groups - 1) * width / 2)
    ax.set_xticklabels(years, rotation=90)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=False))

    if v in ["Wi", "CSC"]:
        ax.set_xlabel("Year")
    if v in ["Wi", "ST"]:
        ax.set_ylabel("Fraction of variance\nexplained by\ninternal variability")
    ax.set_title(var_dict[v])

# figure-level legend (outside right)
fig.legend(
    handles, labels,
    title="Seed",
    loc="center right",
    frameon=False,
    alignment="left",
    bbox_to_anchor=(1.12, 0.5),
)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(pn.figures.get() / "figS6_convergence_seed.jpg", dpi=300)
plt.show()
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
import seaborn as sns
import clt
import pathnavigator
root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\SD6InternalVariability"
pn = pathnavigator.create(root_dir)
pn.code.chdir()

df_sys_all = pd.read_parquet(pn.outputs.ANOVA.get("df_sys_all.parquet"))
#%% Plot ANONA time series
# Define y-variables and hue-variables
y_vars = ['ST', 'Wi', 'CF', 'CSC']
hue_vars = ['Pr', 'Cr', 'Co']

# Get median items for each variable (based on sorted unique values)
def get_median_item(series):
    unique_vals = np.sort(series.unique())
    return unique_vals[len(unique_vals) // 2]

medians = {
    "Pr": get_median_item(df_sys_all["Pr"]),
    "Cr": get_median_item(df_sys_all["Cr"]),
    "Co": get_median_item(df_sys_all["Co"])
}
hue_var_dict = {
    "Pr": "Prec ratio\n(Pr)",
    "Cr": "Crop price\nratio (Cr)",
    "Co": "Corn field\nratio (Co)",
    }

y_var_dict = {
    "ST": "Saturated\nthickness\n(m)",
    "Wi": "Withdrawal\n($10^6 m^3$)",
    "CF": "Field ratio\nfor corn\n(--)",
    "CSC": "Behavioral\nstate changes\n(--)"
    }


# Create 4x3 subplot grid
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(6, 6), sharex=True)

# Iterate through each subplot
for i, y_var in enumerate(y_vars):
    for j, hue_var in enumerate(hue_vars):
        ax = axes[i, j]

        df_subset = df_sys_all.copy()
        for v in hue_vars:
            if v != hue_var:
                df_subset = df_subset[np.isclose(df_subset[v], medians[v])]
        df_subset = df_subset[
            np.isclose(df_subset[hue_var], df_subset[hue_var].min()) |
            np.isclose(df_subset[hue_var], medians[hue_var]) |
            np.isclose(df_subset[hue_var], df_subset[hue_var].max())
            ]

        # Plot
        sns.lineplot(
            data=df_subset,
            x="Year",
            y=y_var,
            hue=hue_var,
            errorbar=("sd", 2),
            err_style="band",
            ax=ax,
            legend=False,
            palette="Set1"
        )

        if i == 0:
            ax.set_title(f'{hue_var_dict[hue_var]}', fontsize=10)
            ax.set_ylim([15, 22])
        if i == 1:
            ax.set_ylim([0, 40])
        if i == 2:
            ax.set_ylim([0, 1.25])
        if i == 3:
            ax.set_ylim([0, 400])

        if j == 0:
            ax.set_ylabel(f"{y_var_dict[y_var]}")
        if j != 0:
            ax.set_ylabel("")
            ax.set_yticklabels([])

        if j != 1:
            ax.set_xlabel("")

        ax.set_xlim([2013, 2022])
fig.align_ylabels()
# Adjust layout
plt.tight_layout()
clt.fig.savefig(fig, filename=pn.figures.get()/"figS5_anova_ts.jpg")
plt.show()
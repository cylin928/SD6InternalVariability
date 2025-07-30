import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pathnavigator

root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\SD6InternalVariability"
pn = pathnavigator.create(root_dir)

#%% Load processed data
df_rank = pd.read_csv(pn.figures.data_for_plotting.get()/"fig3_oat_df_rank.csv", index_col=[0])
df_si = pd.read_csv(pn.figures.data_for_plotting.get()/"fig3_oat_df_si.csv", index_col=[0])

#%% Figure 3 OAT heatmap
factors_ = [
    "Precipitation",
    "Crop price",
    "Initial ratio of\ncorn fields",
    r'Initial well' + '\n' + 'characteristics',#'\n'+r'($\bar{B}$)',
    ]

# Create the figure and axis
fig, ax = plt.subplots(figsize=(3.7, 4.2))

# Plot the heatmap
sns.heatmap(
    df_rank.T[factors_],
    annot=df_si.T[factors_],
    cmap="Oranges_r",# "Reds_r",
    linewidths=0.5,
    fmt=".1f",
    cbar_kws={"label": "Rank of importance"},
    annot_kws={"fontsize": 9},#, "fontweight": "bold"},
    ax=ax,
)

ax.collections[0].colorbar.ax.invert_yaxis()
ax.tick_params(axis="x", labelrotation=90)
ax.tick_params(axis="both", labelsize=10)
# Set titles and labels
# ax.set_title("Heatmap of Ranks Highlighting Top Ranks", fontsize=16)
ax.set_xlabel("Driving factor", fontsize=10)
ax.set_ylabel("Response variable\n(mean of LEMA period)", fontsize=10)

# Adjust layout and display
plt.tight_layout()
fig.savefig(pn.figures.get() / "fig3_oat_heatmap.jpg", dpi=300, bbox_inches='tight')
plt.show()

#%% Figure S3 OAT heatmap
factors_ = [
    "Precipitation",
    "Crop price",
    "Initial ratio of\ncorn fields",
    r'Initial well' + '\n' + 'characteristics',#'\n'+r'($\bar{B}$)',
    "Electricity\nprice",
    "Aquifer\ndrawdown\nrate coef.",
    ]

# Create the figure and axis
fig, ax = plt.subplots(figsize=(5.91, 4.5))

# Plot the heatmap
sns.heatmap(
    df_rank.T[factors_],
    annot=df_si.T[factors_],
    cmap="Oranges_r",# "Reds_r",
    linewidths=0.5,
    fmt=".1f",
    cbar_kws={"label": "Rank of importance"},
    annot_kws={"fontsize": 9},#, "fontweight": "bold"},
    ax=ax,
)

ax.collections[0].colorbar.ax.invert_yaxis()
ax.tick_params(axis="x", labelrotation=90)
ax.tick_params(axis="both", labelsize=10)
ax.set_xlabel("Driving factor", fontsize=10)
ax.set_ylabel("Response variable\n(mean of LEMA period)", fontsize=10)

# Adjust layout and display
plt.tight_layout()
fig.savefig(pn.figures.get() / "figS3_oat_heatmap.jpg", dpi=300, bbox_inches='tight')
plt.show()



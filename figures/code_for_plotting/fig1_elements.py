import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import pathnavigator

root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\SD6InternalVariability"
pn = pathnavigator.create(root_dir)

df = pd.read_csv(pn.data.get("SD6_grid_info_selected.csv"))
# Convert to GeoDataFrame
df["lon"] /= 1.4
df = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df["lon"], df["lat"])])

init = 2011

#%% Plot crop field distribution in 2011
# Plot the points
fig, ax = plt.subplots(figsize=(6, 6))
crop_colors = {"corn": "#EB6E12", "others": "grey"}

for crop, color in crop_colors.items():
    crop_df = df[df[f"Crop{init}"] == crop]
    crop_df.plot(ax=ax, color=color, markersize=50, label=crop)

# # Add title and legend
num_corn = sum(df[f"Crop{init}"] == "corn")
# plt.title(f"# of corn {num_corn}")
# plt.legend()
plt.tight_layout()
#ax.axis("off")

# Make tick labels smaller
ax.tick_params(axis='both', labelsize=14)

# Show only left and bottom spines
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

fig.savefig(pn.figures.get() / "fig1_SD6_corn_dist_2011.jpg", dpi=300, bbox_inches='tight')

plt.show()

#%% Plot dynamics of calibrated model
import pickle
pn.code.chdir()
from plotting import SD6Visual
# Replace the path with the correct location of your .pkl file
with open(pn.models.get('calibrated_model.pkl'), 'rb') as file:
    m = pickle.load(file)

df_sys = pd.read_csv(pn.models.get() / "calibrated_model_df_sys.csv", index_col=[0])
df_agt = pd.read_csv(pn.models.get() / "calibrated_model_df_agt.csv", index_col=[0])

prec_avg = pd.read_csv(pn.data.get("prec_avg_2011_2022.csv"), index_col=[0]).iloc[1:, :]
sd6_data = pd.read_csv(pn.data.get("Data_SD6_2012_2022.csv"), index_col=["year"])

visual = SD6Visual()
visual.output_dir = pn.figures.get()
visual.add_sd6_plotting_info(sd6_data=sd6_data, prec_avg=prec_avg)

# visual.plot_timeseries(df_sys=df_sys, fig_name="")
# visual.plot_crop_ratio(df_sys=df_sys, fig_name="")

visual.plot_st_withdrawal_corn_timeseries(df_sys=df_sys, add_prec_bars_on_secondary_yaxis=False, fig_name="fig1_cali_dynamics.jpg")
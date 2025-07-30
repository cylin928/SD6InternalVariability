#%% Scenarios
import numpy as np
ratio_dict = {
    "prec": np.linspace(0.904, 1.125, 10).round(4),
    "price": np.linspace(0.751, 1.148, 10).round(4),
    "corn": np.linspace(0.720, 0.497, 10).round(4),
}

rng = np.random.default_rng(seed=42)
years = rng.choice(np.arange(2003, 2023), size=(12, 15), replace=True).T
seeds = [1, 2, 67]
# each corn ratio have 5 sequence of samples
# run 3 random seed
scenario_array = []
for pr in ratio_dict["prec"]:
    for p in ratio_dict["price"]:
        for c in ratio_dict["corn"]:
            for ci in range(5):
                for yi in range(len(years)):
                    for seed in seeds:
                        scenario_array.append([pr, p, c, ci, yi, seed])
#["prec_ratio", "price_ratio", "corn_ratio", "corn_i", "boot_i", "seed"]

#%%
import os
import pandas as pd
from tqdm import tqdm
import pathnavigator
root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\SD6InternalVariability"
pn = pathnavigator.create(root_dir)
pn.code.chdir()

def count_state_changes_per_year(df):
    # Initialize a dictionary to store the sum of state changes for each year
    state_changes_by_bid = []

    # Sort the dataframe by year to ensure proper comparisons
    df = df.sort_values("year")

    # Group the dataframe by year
    grouped = df.groupby("year")

    # Iterate over the grouped years
    previous_group = None
    for year, current_group in grouped:
        # Reset index for ease of calculation
        current_group = current_group.reset_index(drop=True)

        # Initialize the change counter for this year
        state_change_sum = 0

        if previous_group is not None:
            # Inner merge to align by `bid` between the current and previous year
            merged = current_group.merge(
                previous_group, on="bid", suffixes=("_current", "_previous")
            )

            # Calculate the state change by comparing states between years
            state_change_sum = sum(merged["state_current"] != merged["state_previous"])

        # Store the sum of state changes for the current year
        state_changes_by_bid.append(state_change_sum)
        # state_changes_by_bid[year] = [state_change_sum]

        # Update the previous group for the next iteration
        previous_group = current_group
    # dff = pd.DataFrame(state_changes_by_bid)

    return state_changes_by_bid

#%%
df_dir = rf"C:\Users\{pathnavigator.user}\Documents\ANOVA"
# files = os.listdir(df_dir)
# files_sys = [f for f in files if "df_sys" in f]
# files_agt = [f for f in files if "df_agt" in f]

def read_df_sys(scenario=None, df_dir=df_dir, usecols=['year', 'GW_st', 'withdrawal', "rainfed", "corn", "others"]):
    scen_name = "-".join([str(x) for x in scenario])
    path_sys = os.path.join(df_dir, f"df_sys-{scen_name}.csv")
    path_agt = os.path.join(df_dir, f"df_agt-{scen_name}.csv")

    #scen_name_list = ["Prec ratio", "Crop price ratio", "Corn ratio", "Corn dist.", "Realization", "Seed"]
    scen_name_list = ["Pr", "Cr", "Co", "Cd", "Re", "Seed"]
    df_agt = pd.read_csv(path_agt)
    df_sys = pd.read_csv(path_sys, usecols=usecols)
    df_sys["consumat_state_changes"] = count_state_changes_per_year(df_agt)
    df_sys["total_profit"] = df_agt.groupby("year")[["year", "profit"]].sum()["profit"].values
    df_sys["withdrawal"] /= 100  # 10^4 to 10^6 m3
    df_sys.rename(
        #columns={"year": "Year", "GW_st": "Saturated thickness", "withdrawal": "Withdrawal"},
        columns={
            "year": "Year",
            "GW_st": "ST",
            "withdrawal": "Wi",
            "rainfed": "RF",
            "corn": "CF",
            "others": "OF",
            "consumat_state_changes": "CSC",
            "total_profit": "TP"
            },
        inplace=True
        )
    for c, v in zip(scen_name_list, scenario):
        df_sys[c] = v
    return df_sys


#dfs_sys = [read_df_sys(f) for f in files_sys[0:3000]]
#dfs_sys = [read_df_sys(f) for f in tqdm(files_sys)]
dfs_sys = []
for i, scenario in tqdm(enumerate(scenario_array)):
    try:
        df = read_df_sys(scenario)
        dfs_sys.append(df)
    except Exception as e:
        print(f"Skipping scenario {scenario} due to error: {e}")
        continue
# We want to organize the df to contain columns
# Y: year / Withdrawal / Saturated thickness
# X: "Prec ratio", "Crop price ratio", "Corn ratio", "Environmental variability"

df_sys_all = pd.concat(dfs_sys).reset_index(drop=True)
#df_sys_all["Environmental variability"] = df_sys_all.groupby(["Corn dist.", "Realization"]).ngroup()
# df_sys_all["PrCr"] = df_sys_all.groupby(["Pr", "Cr"]).ngroup()
# df_sys_all["CrCo"] = df_sys_all.groupby(["Cr", "Co"]).ngroup()
# df_sys_all["CoPr"] = df_sys_all.groupby(["Co", "Pr"]).ngroup()
# df_sys_all["PrCrCo"] = df_sys_all.groupby(["Pr", "Cr", "Co"]).ngroup()

df_sys_all.to_csv(pn.outputs.ANOVA.get()/"df_sys_all.csv", index=False)
import os
import re
import pandas as pd
import pathnavigator

root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\SD6InternalVariability"
pn = pathnavigator.create(root_dir)

#%% Process output data for plotting (no need to rerun)
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
    return state_changes_by_bid

def extract_oat(scen, exp_folder):
    df_agt = pd.read_csv(pn.outputs.get(f"{exp_folder}/df_agt-{scen}.csv"))
    df = pd.read_csv(pn.outputs.get(f"{exp_folder}/df_sys-{scen}.csv"))
    df["state_changes"] = count_state_changes_per_year(df_agt)
    df["profit"] = df_agt.groupby("year")[["year", "profit"]].sum()["profit"].values
    df = df.iloc[1:, :]
    df = df.set_index("year")
    dff = df.mean()
    number_list = [float(num) for num in re.findall(r"[+]?\d*\.\d+|\d+", scen)]
    columns = [
        "prec_ratio",
        "crop_price_ratio",
        "elec_price_ratio",
        "B_mean_ratio",
        "aq_a_ratio",
        "corn_adj_ratio",
    ]
    additional_data = pd.Series(number_list, index=columns)
    dff = pd.concat([dff, additional_data])
    dff["scen"] = scen
    return dff

def cal_df_rank_and_df_si(exp_folder):
    flist = os.listdir(pn.outputs.get(f"{exp_folder}"))
    scen_list = list(set([s[7:-4] for s in flist]))
    df_oat = []
    for scen in scen_list:
        df_oat.append(extract_oat(scen, exp_folder))

    df_oat = pd.concat(df_oat, axis=1).T

    vars = ["GW_st", "withdrawal", "corn", "others", "state_changes", "profit", "rainfed"]
    vars_ = [
        "Saturated\nthickness",
        "Withdrawal",
        "Field ratio\nfor corn",
        "Field ratio\nfor others",
        "Behavioral\nstate changes",
        "Total profit",
        "Rainfed\nfield ratio",
    ]
    factors = [
        "Precipitation",
        "Crop price",
        "Electricity\nprice",
        r'Initial well' + '\n' + 'characteristics',#'\n'+r'($\bar{B}$)',
        "Aquifer\ndrawdown\nrate coef.",
        "Initial ratio of\ncorn fields",
    ]

    df_base = df_oat[df_oat["scen"] == "1.0-1.0-1.0-1.0-1.0-1.0"]
    df_oat_pct = df_oat.copy()
    df_oat_pct.loc[:, vars] = (
        (df_oat.loc[:, vars] - df_base.loc[:, vars].iloc[0])
        / df_base.loc[:, vars].iloc[0]
        * 100
    )
    import clt
    clt.fig.size.erl
    def form_scen_list(ratio):
        l = []
        for i in range(6):
            ll = [1.0] * 6
            ll[i] = ratio
            ll = "-".join([f"{x:.1f}" for x in ll])
            l.append(ll)
        return l


    def get_rank_df(ratio, data):
        l = form_scen_list(ratio)
        selected_df = data[data["scen"].isin(l)]
        selected_df = selected_df.set_index("scen", drop=True).loc[l, vars]
        selected_df.columns = vars_
        selected_df.index = factors
        selected_df_abs = selected_df.abs()

        ranked_df = selected_df_abs.copy()
        for col in selected_df_abs.columns:
            ranked_df[col] = selected_df_abs[col].rank(method="dense", ascending=False)
        return selected_df, ranked_df


    def sort_by_row_mean(df):
        df["average_rank"] = df.mean(axis=1)
        df = df.sort_values(by="average_rank", ascending=True)
        df = df.drop(columns=["average_rank"])
        return df

    selected_dfs = []
    rank_dfs = []
    ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5]
    for ratio in ratios:
        d1, d2 = get_rank_df(ratio, df_oat_pct)
        selected_dfs.append(d1)
        rank_dfs.append(d2)
    mean_ranked_df = pd.concat(rank_dfs).groupby(level=0).mean()
    mean_ranked_df = sort_by_row_mean(mean_ranked_df)
    std_selected_df = pd.concat(selected_dfs).groupby(level=0).std()
    std_selected_df = std_selected_df.loc[mean_ranked_df.index, mean_ranked_df.columns]

    return mean_ranked_df, std_selected_df

dfs_rank = []
dfs_si = []
for seed in [1,2,67]:
    exp_folder = f"OAT_{seed}"
    mean_ranked_df, std_selected_df = cal_df_rank_and_df_si(exp_folder)
    dfs_rank.append(mean_ranked_df)
    dfs_si.append(std_selected_df)

df_rank = sum(dfs_rank) / len(dfs_rank)
df_si = sum(dfs_si) / len(dfs_si)
df_rank.to_csv(pn.figures.data_for_plotting.get()/"fig3_oat_df_rank.csv")
df_si.to_csv(pn.figures.data_for_plotting.get()/"fig3_oat_df_si.csv")


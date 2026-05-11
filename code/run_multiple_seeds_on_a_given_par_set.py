
import os
import sys
import dill
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pathnavigator
sys.setrecursionlimit(10000)  # Set to a higher value for dill deep dict.
root_dir = rf"/Users/{pathnavigator.user}/Documents/GitHub/SD6InternalVariability"
pn = pathnavigator.create(root_dir)
pn.code.chdir()
from py_champ.models.sd6_model_1f1w import SD6Model4SingleFieldAndWell
from utils import (
    load_x_from_cali_txt_output,
    cal_rmse_metrices, 
    cal_weighted_rmse, 
    normalize_st_withdrawal, 
    update_model_inputs_from_cali_x
    )
from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
set_loky_pickler("dill")


# Set shortcut
pn.inputs.set_sc("input_pkl", "Inputs_SD6_2012_2022.pkl")
pn.models.set_sc("cali_x", "calibrated_parameters.txt")
pn.data.set_sc("sd6_data", "Data_SD6_2012_2022.csv")

#%%
# Load inputs
with open(pn.sc.input_pkl, "rb") as f:
    inputs = dill.load(f)


# Load calibrated parameters
cali_x = load_x_from_cali_txt_output(pn.sc.cali_x)

# Load sd6 data
sd6_data = pd.read_csv(pn.sc.sd6_data, index_col=["year"])
sd6_data = normalize_st_withdrawal(sd6_data)

def run_single_simulation(cali_x, seed, inputs, sd6_data, exp_dir):
    import os
    import sys
    import pathnavigator
    sys.setrecursionlimit(10000)  # Set to a higher value for dill deep dict.
    root_dir = rf"/Users/{pathnavigator.user}/Documents/GitHub/SD6InternalVariability"
    pn = pathnavigator.create(root_dir)
    pn.code.chdir()
    pn.code.add_to_sys_path()
    from py_champ.models.sd6_model_1f1w import SD6Model4SingleFieldAndWell
    from utils import (
        cal_rmse_metrices, 
        normalize_st_withdrawal, 
        update_model_inputs_from_cali_x
        )
    # Prepare variables for simulation
    crop_options = ["corn", "others"]
    init_year = 2011
    #seed = 67

    (aquifers_dict, fields_dict, wells_dict, finances_dict, behaviors_dict, prec_aw_step, crop_price_step) = inputs
    fields_dict, crop_price_step, pars = update_model_inputs_from_cali_x(cali_x, fields_dict, crop_price_step)
    try:
        m = SD6Model4SingleFieldAndWell(
            pars=pars,
            crop_options=crop_options,
            prec_aw_step=prec_aw_step,
            aquifers_dict=aquifers_dict,
            fields_dict=fields_dict,
            wells_dict=wells_dict,
            finances_dict=finances_dict,
            behaviors_dict=behaviors_dict,
            crop_price_step=crop_price_step,
            init_year=init_year,
            end_year=2022,
            lema_options=(True, "wr_LEMA_5yr", 2013),
            show_step=False,
            show_initialization=False,
            seed=seed,
            gurobi_dict = {"LogToConsole": 0, "NonConvex": 2, "Presolve": -1}
        )

        for i in range(11):
            m.step()
        m.end()
        df_sys, _ = m.get_dfs(m)
        
        # Normalize GW_st withdrawal to [0, 1] according to obv (i.e., data)
        df_sys = df_sys.loc[2012 :]
        df_sys = normalize_st_withdrawal(df_sys)
        
        # Calculate metrices
        metrices = cal_rmse_metrices(df_sys, sd6_data)
        metrices.to_csv(os.path.join(exp_dir, f"metrices_seed_{seed}.csv"))
        print(metrices)
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return e

#%%
df_sys = pd.read_csv("/Users/cl/Documents/GitHub/SD6InternalVariability/outputs/OAT_67/df_sys-1.0-1.0-1.0-1.0-1.0-1.0.csv",
                     index_col=["year"])

df_sys = df_sys.loc[2012 :]
df_sys["GW_st"] = (df_sys["GW_st"] - 17.5577) / (18.2131 - 17.5577)
df_sys["withdrawal"] = (df_sys["withdrawal"] - 1310.6749) / (
    3432.4528 - 1310.6749
)

sd6_data = pd.read_csv(pn.sc.sd6_data, index_col=["year"])
sd6_data["GW_st"] = (sd6_data["GW_st"] - 17.5577) / (18.2131 - 17.5577)
sd6_data["withdrawal"] = (sd6_data["withdrawal"] - 1310.6749) / (
    3432.4528 - 1310.6749
)
sd6_data = sd6_data.loc[2012 :]





df_sys = normalize_st_withdrawal(df_sys)
metrices = cal_rmse_metrices(df_sys, sd6_data)

#%% Run multiple seeds
pn.outputs.mkdir("run_multiple_seeds_on_cali_x")
exp_dir = pn.outputs.get("run_multiple_seeds_on_cali_x")
seeds = list(range(1, 201)) # 1 - 200 seeds
os.cpu_count()
logs = Parallel(n_jobs=8, verbose=10)(
    delayed(run_single_simulation)(
        cali_x=cali_x,
        seed=seed,
        inputs=inputs,
        sd6_data=sd6_data,
        exp_dir=exp_dir
    )
    for seed in seeds
)
print("All simulations are completed")
print(logs)

#%% Analyze and plots
exp_dir = pn.outputs.get("run_multiple_seeds_on_cali_x")
metric_files = sorted(
    f for f in os.listdir(exp_dir)
    if f.startswith("metrices_seed_") and f.endswith(".csv")
)

met_frames = []
for f in metric_files:
    met = pd.read_csv(os.path.join(exp_dir, f), index_col=[0])
    fitness_rows = ["GW_st", "withdrawal", "corn", "others"]
    rows_present = [r for r in fitness_rows if r in met.index]
    if rows_present:
        met.loc["fitness"] = met.loc[rows_present].mean(axis=0)
    else:
        met.loc["fitness"] = float("nan")
    met["file_name"] = f
    met_frames.append(met)

met_all = pd.concat(met_frames, axis=0, ignore_index=False)

print(f"Loaded {len(metric_files)} metrics files.")

# Use index labels as x-axis groups and the first three columns as hue groups.
plot_df = met_all.iloc[:, :3].copy()
plot_df["index_group"] = met_all.index

plot_long = plot_df.melt(
    id_vars="index_group",
    var_name="hue_group",
    value_name="value",
)

plt.figure(figsize=(12, 6))
sns.boxplot(data=plot_long, x="index_group", y="value", hue="hue_group")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

#%% Plot only fitness values across files.
fitness_df = met_all.loc[met_all.index == "fitness"].copy()
if not fitness_df.empty:
    fitness_plot_df = fitness_df.iloc[:, :3].copy()
    fitness_plot_df["file_name"] = fitness_df["file_name"].values
    fitness_long = fitness_plot_df.melt(
        id_vars="file_name",
        var_name="metric",
        value_name="value",
    )

    metric_palette = {
        metric: color
        for metric, color in zip(
            fitness_long["metric"].unique(),
            sns.color_palette("Set2", n_colors=fitness_long["metric"].nunique()),
        )
    }

    plt.figure(figsize=(5, 4))
    sns.boxplot(data=fitness_long, x="metric", y="value", palette=metric_palette)
    
    plt.xlabel("Period", fontsize=12)
    plt.ylabel("$RMSE$", fontsize=12)
    plt.ylim([0,1])
    plt.tight_layout()
    plt.show()
    
else:
    print("No fitness rows found in met_all.")

###############################################################################
#%% RMSE of the final calibrated model spread
pn.outputs.mkdir("run_multiple_seeds_on_cali_x_median")
exp_dir = pn.outputs.get("run_multiple_seeds_on_cali_x_median")
seeds = [3, 56, 67, 89, 123]
os.cpu_count()

with open(pn.outputs.sd6_cali2.get("PSO_it100.pkl"), "rb") as f:
    load_dict = dill.load(f)

sw = load_dict["swarm"]
cali_x_median = sw.best_pos

logs = Parallel(n_jobs=5, verbose=10)(
    delayed(run_single_simulation)(
        cali_x=cali_x_median,
        seed=seed,
        inputs=inputs,
        sd6_data=sd6_data,
        exp_dir=exp_dir
    )
    for seed in seeds
)
print("All simulations are completed")
print(logs)

#%% Analyze and plots
exp_dir = pn.outputs.get("run_multiple_seeds_on_cali_x_median")
metric_files = sorted(
    f for f in os.listdir(exp_dir)
    if f.startswith("metrices_seed_") and f.endswith(".csv")
)

met_frames = []
for f in metric_files:
    met = pd.read_csv(os.path.join(exp_dir, f), index_col=[0])
    fitness_rows = ["GW_st", "withdrawal", "corn", "others"]
    rows_present = [r for r in fitness_rows if r in met.index]
    if rows_present:
        met.loc["fitness"] = met.loc[rows_present].mean(axis=0)
    else:
        met.loc["fitness"] = float("nan")
    met["file_name"] = f
    met_frames.append(met)

met_all = pd.concat(met_frames, axis=0, ignore_index=False)

print(f"Loaded {len(metric_files)} metrics files.")

#%%
fitness_df = met_all.loc[met_all.index == "fitness"].copy()


fitness_df["cali"]

# Simple bar plot for calibration RMSE

plt.figure(figsize=(5, 4))

sns.barplot(
    x=fitness_df["file_name"],
    y=fitness_df["cali"],
)

plt.xlabel("Seed", fontsize=12)
plt.ylabel(r"$RMSE (2013-2019)$", fontsize=12)

# Optional: show only seed number on x-axis
seed_labels = (
    fitness_df["file_name"]
    .str.extract(r"seed_(\d+)")[0]
)

plt.xticks(
    ticks=range(len(seed_labels)),
    labels=seed_labels,
    rotation=0,
)

plt.tight_layout()
plt.show()
#%%
fitness_df = met_all.loc[met_all.index == "fitness"].copy()

if not fitness_df.empty:
    fitness_plot_df = fitness_df.iloc[:, :3].copy()
    fitness_plot_df["file_name"] = fitness_df["file_name"].values

    # Extract only seed number from filename
    fitness_plot_df["seed"] = (
        fitness_plot_df["file_name"]
        .str.extract(r"seed_(\d+)")
        .astype(str)
    )

    fitness_long = fitness_plot_df.melt(
        id_vars=["file_name", "seed"],
        var_name="metric",
        value_name="value",
    )

    # One color per seed
    seed_palette = {
        seed: color
        for seed, color in zip(
            fitness_long["seed"].unique(),
            sns.color_palette("Set2", n_colors=fitness_long["seed"].nunique()),
        )
    }

    plt.figure(figsize=(5, 4))

    sns.stripplot(
        data=fitness_long,
        x="metric",
        y="value",
        hue="seed",
        palette=seed_palette,
        dodge=True,
        size=8,
        alpha=0.9,
    )

    plt.xlabel("Period", fontsize=12)
    plt.ylabel(r"$RMSE$", fontsize=12)
    plt.ylim([0, 1])

    plt.legend(
        title="Seed",
        #bbox_to_anchor=(1.05, 1),
        loc="upper right",
        ncols=3,
        frameon=True,
    )

    plt.tight_layout()
    plt.show()

else:
    print("No fitness rows found in met_all.")




import os
import dill
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathnavigator import PathNavigator

root_dir = os.path.expanduser("~/Github/EnvHeteroImpact")
root_dir = r"C:\Users\CL\Documents\GitHub\EnvHeteroImpactGW"
pn = PathNavigator(root_dir)

pn.inputs.set_sc("input_pkl", "Inputs_SD6_2012_2022.pkl")

with open(pn.sc.input_pkl, "rb") as f:
    inputs = dill.load(f)

prec_avg = pd.read_csv(pn.data.get("prec_avg_2003_2022.csv"), index_col=[0]).iloc[1:, :]
prec_avg["year"] = prec_avg.index

kfma_income = pd.read_csv(pn.data.get("KFMA_crop_income.csv"), index_col="Year")
kfma_income["others"] = (
    kfma_income["sorghum"] * 194.0593
    + kfma_income["soybeans"] * 146.3238
    + kfma_income["wheat"] * 141.1518
) / (194.0593 + 146.3238 + 141.1518)
# kfma_income.plot(xlabel="Year", ylabel="$/bu", xlim=[2003, 2022]).legend(ncol=2, frameon=False)
kfma_income = kfma_income.loc[2003:2022, ["corn", "others"]].round(3)
kfma_income.plot(xlabel="Year", ylabel="$/bu", xlim=[2003, 2022])
plt.show()
kfma_income["prec"] = prec_avg["annual"]
kfma_income.corr().round(2)

fig, ax = plt.subplots()
ax.scatter(kfma_income["corn"], kfma_income["prec"], label="Corn")
ax.scatter(kfma_income["others"], kfma_income["prec"], label="Others")
ax.set_xlabel("Precipitation (cm)")
ax.set_ylabel("$/bu")
ax.legend()
plt.show()


# %%
def plot_hist(samples, data=None, statistic="mean", prefix=""):
    means = np.mean(samples, axis=0)
    stds = np.std(samples, axis=0)
    fig, ax = plt.subplots()
    if statistic == "mean":
        ax.hist(means, bins=50, edgecolor="black", alpha=0.7)
        if data is not None:
            ax.axvline(np.mean(data), color="red")
        else:
            ax.axvline(1, color="red")
    else:
        ax.hist(stds, bins=50, edgecolor="black", alpha=0.7)
        if data is not None:
            ax.axvline(np.std(data), color="red")
        else:
            ax.axvline(0, color="red")
    ax.set_xlabel(prefix + statistic)
    ax.set_ylabel("Frequency")
    plt.show()


# Precipitation
samples = np.random.choice(prec_avg["annual"], size=(11, 10000), replace=True, p=None)
means = np.mean(samples, axis=0)
stds = np.std(samples, axis=0)

plot_hist(samples, prec_avg["annual"], statistic="mean", prefix="Prec ")
plot_hist(samples, prec_avg["annual"], statistic="std", prefix="Prec ")
plot_hist(samples / means, None, statistic="mean", prefix="Prec ")
plot_hist(samples / means, None, statistic="std", prefix="Prec ")

samples = np.random.choice(kfma_income["corn"], size=(11, 10000), replace=True, p=None)
means = np.mean(samples, axis=0)
stds = np.std(samples, axis=0)
plot_hist(samples, kfma_income["corn"], statistic="mean", prefix="Corn ")
plot_hist(samples, kfma_income["corn"], statistic="std", prefix="Corn ")
plot_hist(samples / means, None, statistic="mean", prefix="Corn ")
plot_hist(samples / means, None, statistic="std", prefix="Corn ")

samples = np.random.choice(
    kfma_income["others"], size=(11, 10000), replace=True, p=None
)
means = np.mean(samples, axis=0)
stds = np.std(samples, axis=0)
plot_hist(samples, kfma_income["others"], statistic="mean", prefix="Others ")
plot_hist(samples, kfma_income["others"], statistic="std", prefix="Others ")
plot_hist(samples / means, None, statistic="mean", prefix="Others ")
plot_hist(samples / means, None, statistic="std", prefix="Others ")


# %%
def quantile_of_boostrapping_means(data):
    samples = np.random.choice(data, size=(12, 1_000_000), replace=True, p=None)
    means = np.mean(samples, axis=0)
    quantile_5 = np.quantile(means, 0.05).round(2)
    quantile_95 = np.quantile(means, 0.95).round(2)
    return quantile_5, quantile_95


quantile_of_boostrapping_means(prec_avg["annual"])
# (np.float64(48.29), np.float64(60.05))
prec_avg["annual"][7:].mean()
# np.float64(53.39131481481482)
# 0.904 1.125


quantile_of_boostrapping_means(kfma_income["corn"])
# (np.float64(3.93), np.float64(5.83))
kfma_income["corn"][7:].mean()
# np.float64(5.231230769230769)
# 0.751 1.115
quantile_of_boostrapping_means(kfma_income["others"])
# (np.float64(6.71), np.float64(8.37))
kfma_income["others"][7:].mean()
# np.float64(7.29023076923077)
# 0.920 1.148
# => 0.751 1.148

crops = pd.read_csv(pn.data.get("SD6_grid_info_selected.csv"))[
    [f"Crop{y}" for y in range(2006, 2023)]
]
corn_ratios = crops.apply(lambda col: (col == "corn").mean()).values
max(corn_ratios)
# np.float64(0.7202380952380952)
min(corn_ratios)
# np.float64(0.49702380952380953)

# %%
ratio_dict = {
    "prec": np.linspace(0.904, 1.125, 10).round(4),
    "price": np.linspace(0.751, 1.148, 10).round(4),
    "corn": np.linspace(0.720, 0.497, 10).round(4),
}

rng = np.random.default_rng(seed=42)
years = rng.choice(np.arange(2011, 2023), size=(12, 15), replace=True)
seeds = [1, 2, 67]
# each corn ratio have 5 sequence of samples
# run 3 random seed
scenarios = []
for pr in ratio_dict["prec"]:
    for p in ratio_dict["price"]:
        for c in ratio_dict["corn"]:
            for ci in range(5):
                for yi in range(len(years)):
                    for seed in seeds:
                        scenarios.append([pr, p, c, ci, yi, seed])
print("Number of scenarios: ", len(scenarios))
df_years = pd.DataFrame(years)
df_years.to_csv(pn.scenarios.get() / "exp4_years.csv")










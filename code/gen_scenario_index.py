import os
import sys

import pandas as pd
import numpy as np

wd = rf"C:/Users/{os.getlogin()}/Documents/GitHub/SD6InternalVariability"
if wd not in sys.path:
    sys.path.append(os.path.join(wd, "code"))

from utils import ProjPaths

paths = ProjPaths(wd)
paths.add_subfolder("scenarios")
# General information

# 00 means the baseline


# we will dynamically compute the ratios
# ratios = list(np.arange(0.01, 1, 0.01).round(2))


# %%
# Create scenario index for the task 0 (baseline). [6]
py_seeds = [67, 56, 78, 90, 12, 34, 47]

prefix = "t0"
count = 0
df = {}
for py_seed in py_seeds:
    count += 1
    idx = f"{prefix}_{count:08d}"
    df[f"R00_C00_S00_P00_E00_Pseed{py_seed:02d}"] = [
        idx,
        "R00",
        "C00",
        "S00",
        0,
        "P00",
        "E00",
        py_seed,
    ]
    # [idx, R00, C00, S00, 0, P00, E00, XX (6)]
df = pd.DataFrame.from_dict(
    df,
    orient="index",
    columns=["idx", "range", "ratio", "set", "r_seed", "prec_id", "eco_id", "py_seed"],
)
df.to_csv(
    os.path.join(paths.scenarios, f"scenario_index_{prefix}.csv"),
    index=True,
    index_label="scenario_name",
)
# %%
# Create scenario index for the task 2. [2328]
# How initial crop heterogeniety affects the model outcomes?
py_seeds = [67, 56, 78, 90, 12, 34, 47]
ranges = ["R03", "R07", "R11", "R15", "R19", "R24", "R28"]

prefix = "t2"
count = 0
df = {}
set_ = "S01"
r_seed = 3256
for range_ in ranges:
    # Load ratios from generated scenarios
    # ratio = CXX
    ratios = list(
        pd.read_csv(
            os.path.join(paths.scenarios, f"{range_}_{set_}_Rseed{r_seed}.csv"),
            index_col=[0],
        ).columns
    )
    for ratio in ratios:
        for py_seed in py_seeds:
            count += 1
            idx = f"{prefix}_{count:08d}"
            df[f"{range_}_{ratio}_{set_}_P00_E00_Pseed{py_seed:02d}"] = [
                idx,
                range_,
                ratio,
                set_,
                r_seed,
                "P00",
                "E00",
                py_seed,
            ]
            # [idx, RXX, CXX, S01, 3256, P00, E00, XX]
df = pd.DataFrame.from_dict(
    df,
    orient="index",
    columns=["idx", "range", "ratio", "set", "r_seed", "prec_id", "eco_id", "py_seed"],
)
df.to_csv(
    os.path.join(paths.scenarios, f"scenario_index_{prefix}.csv"),
    index=True,
    index_label="scenario_name",
)

# %%
# Create scenario index for the task 3. [6000]
# How climate and economic uncertainty play a role?
py_seeds = [67, 56, 78, 90, 12, 34, 47]
sets = [f"S{i:02d}" for i in range(1, 11)]
prec_ids = ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09", "P10"]
eco_ids = ["E01", "E02", "E03", "E04", "E05", "E06", "E07", "E08", "E09", "E10"]

prefix = "t3"
count = 0
df = {}
r_seed = 3256

for set_ in sets:
    for prec_id in prec_ids:
        for eco_id in eco_ids:
            for py_seed in py_seeds:
                count += 1
                idx = f"{prefix}_{count:08d}"
                df[f"R00_C00_{set_}_{prec_id}_{eco_id}_Pseed{py_seed:02d}"] = [
                    idx,
                    "R00",
                    "C00",
                    set_,
                    r_seed,
                    prec_id,
                    eco_id,
                    py_seed,
                ]
                # [idx, R00, C00, SXX, 3256, PXX, EXX, XX]

df = pd.DataFrame.from_dict(
    df,
    orient="index",
    columns=["idx", "range", "ratio", "set", "r_seed", "prec_id", "eco_id", "py_seed"],
)
df.to_csv(
    os.path.join(paths.scenarios, f"scenario_index_{prefix}.csv"),
    index=True,
    index_label="scenario_name",
)

# %%
n = 20
sets = [f"S{i:02d}" for i in range(1, n + 1)]
prec_ids = [f"P{i:02d}" for i in range(1, n + 1)]
eco_ids = [f"E{i:02d}" for i in range(1, n + 1)]

prefix = "t3v2"
count = 0
df = {}
sets = [f"S{i:02d}" for i in range(1, 11)]
r_seed = 3256

for set_ in sets:
    for prec_id in prec_ids:
        for eco_id in eco_ids:
            count += 1
            idx = f"{prefix}_{count:08d}"
            df[f"R00_C00_{set_}_{prec_id}_{eco_id}_Pseed{py_seed:02d}"] = [
                idx,
                "R00",
                "C00",
                set_,
                r_seed,
                prec_id,
                eco_id,
                67,
            ]
            # [idx, R00, C00, SXX, 3256, PXX, EXX, XX]

df = pd.DataFrame.from_dict(
    df,
    orient="index",
    columns=["idx", "range", "ratio", "set", "r_seed", "prec_id", "eco_id", "py_seed"],
)
df.to_csv(
    os.path.join(paths.scenarios, f"scenario_index_{prefix}.csv"),
    index=True,
    index_label="scenario_name",
)

# %%
# Create scenario index for the task 4. []
# Try to generalize the successful sd6 by quantifying the climate and economic
# uncertainty.
n = 10
prec_ids = [f"P{i:02d}" for i in range(1, n + 1)]
eco_ids = [f"E{i:02d}" for i in range(1, n + 1)]

prefix = "t4"
count = 0
df = {}
py_scen_seed = 7834
df_rc_samples = pd.read_csv(
    os.path.join(paths.scenarios, f"rc_samples_{py_scen_seed}.csv")
)
df_rc_samples.dropna(inplace=True)
r_seed = 3256
py_seed = 67

for prec_id in prec_ids:
    for eco_id in eco_ids:
        for rc_idx in df_rc_samples["rc_idx"]:
            count += 1
            idx = f"{prefix}_{count:08d}"
            df[f"{rc_idx}_{prec_id}_{eco_id}_Pseed{py_seed:02d}"] = [
                idx,
                rc_idx,
                r_seed,
                prec_id,
                eco_id,
                py_seed,
                py_scen_seed,
            ]
            # [idx, RCXXXX, XXXX, PXX, EXX, XX, XXXX]

df = pd.DataFrame.from_dict(
    df,
    orient="index",
    columns=["idx", "rc_idx", "r_seed", "prec_id", "eco_id", "py_seed", "py_scen_seed"],
)
df.to_csv(
    os.path.join(paths.scenarios, f"scenario_index_{prefix}.csv"),
    index=True,
    index_label="scenario_name",
)

# %%
n = 10
prec_ids = [f"P{i:02d}" for i in range(1, n + 1)]
eco_ids = [f"E{i:02d}" for i in range(1, n + 1)]

prefix = "t4"
count = 21201
df = {}
py_scen_seed = 3924
df_rc_samples = pd.read_csv(
    os.path.join(paths.scenarios, f"rc_samples_{py_scen_seed}.csv")
)
df_rc_samples.dropna(inplace=True)
r_seed = 3256
py_seed = 67

for prec_id in prec_ids:
    for eco_id in eco_ids:
        for rc_idx in df_rc_samples["rc_idx"]:
            count += 1
            idx = f"{prefix}_{count:08d}"
            df[f"{rc_idx}_{prec_id}_{eco_id}_Pseed{py_seed:02d}"] = [
                idx,
                rc_idx,
                r_seed,
                prec_id,
                eco_id,
                py_seed,
                py_scen_seed,
            ]
            # [idx, RCXXXX, XXXX, PXX, EXX, XX, XXXX]

df = pd.DataFrame.from_dict(
    df,
    orient="index",
    columns=["idx", "rc_idx", "r_seed", "prec_id", "eco_id", "py_seed", "py_scen_seed"],
)
prefix_ = prefix + "v2"
df.to_csv(
    os.path.join(paths.scenarios, f"scenario_index_{prefix_}.csv"),
    index=True,
    index_label="scenario_name",
)

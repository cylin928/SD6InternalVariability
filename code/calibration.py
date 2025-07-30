import os
import numpy as np
from py_champ.utility.util import TimeRecorder
from py_champ.models.particle_swarm import GlobalBestPSO


def obj_func(x, seeds, exp_dir, **kwargs):
    # Need to be placed here to use dill in joblib
    import os
    import sys
    import dill
    import numpy as np
    import pandas as pd
    from py_champ.models.sd6_model_1f1w import SD6Model4SingleFieldAndWell

    sys.setrecursionlimit(10000)  # Set to a higher value for dill deep dict.
    wd = rf"C:/Users/{os.getlogin()}/Documents/GitHub/SD6InternalVariability"
    if wd not in sys.path:
        sys.path.append(os.path.join(wd, "code"))

    from utils import ProjPaths

    # Add file paths
    paths = ProjPaths(wd)

    # Add a custom path to save the experiment outputs (e.g., models, results)
    paths.add_other_path("exp_dir", exp_dir)

    init_year = 2011
    paths.add_file(f"Inputs_SD6_{init_year+1}_2022.pkl", "inputs", "input_pkl")
    paths.add_file(f"prec_avg_{init_year}_2022.csv", "data", "prec_avg")
    paths.add_file(f"Data_SD6_{init_year+1}_2022.csv", "data", "sd6_data")

    # Load inputs
    with open(paths.input_pkl, "rb") as f:
        (
            aquifers_dict,
            fields_dict,
            wells_dict,
            finances_dict,
            behaviors_dict,
            prec_aw_step,
            crop_price_step,
        ) = dill.load(f)

    cali_years = 7
    warmup_years = 2  # 2011-2012

    ### Load observation data
    sd6_data = pd.read_csv(paths.sd6_data, index_col=["year"])
    # Normalize GW_st withdrawal to [0, 1] according to obv
    sd6_data["GW_st"] = (sd6_data["GW_st"] - 17.5577) / (18.2131 - 17.5577)
    sd6_data["withdrawal"] = (sd6_data["withdrawal"] - 1310.6749) / (
        3432.4528 - 1310.6749
    )
    sd6_data = sd6_data.loc[
        init_year + warmup_years : init_year + warmup_years + cali_years - 1
    ]

    crop_options = ["corn", "others"]

    ### Read PSO variables
    i_iter = kwargs.get("i_iter")
    i_particle = kwargs.get("i_particle")

    ### Setup calibrated parameters
    for fid in fields_dict:
        fields_dict[fid]["water_yield_curves"]["others"] = [
            x[0],
            x[1],
            x[2],
            x[3],
            x[4],
            0.1186,
        ]
    for yr in crop_price_step["finance"]:
        crop_price_step["finance"][yr]["others"] *= x[5]

    pars = {
        "perceived_risk": x[6],
        "forecast_trust": x[7],
        "sa_thre": x[8],
        "un_thre": x[9],
    }

    rmse_list = []
    for seed in seeds:
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
                seed=seed,
            )

            m.particles = x
            for _ in range(cali_years + warmup_years - 1):  # 2011-2019
                m.step()
            m.end()  # despose gurobi env

            # Output dfs
            df_sys, _ = m.get_dfs(m)

            # Normalize GW_st withdrawal to [0, 1] according to obv (i.e., data)
            df_sys = df_sys.loc[init_year + warmup_years :]
            df_sys["GW_st"] = (df_sys["GW_st"] - 17.5577) / (18.2131 - 17.5577)
            df_sys["withdrawal"] = (df_sys["withdrawal"] - 1310.6749) / (
                3432.4528 - 1310.6749
            )

            # Calculate metrices
            metrices = m.get_metrices(df_sys, sd6_data)

            # Calculate obj
            rmse_sys = metrices.loc[["GW_st", "withdrawal"], "rmse"].mean()
            rmse_crop = metrices.loc[crop_options, "rmse"].mean()
            rmse = (rmse_sys + rmse_crop) / 2
            m.rmse = rmse
            # Save the model
            with open(
                os.path.join(
                    exp_dir,
                    f"{int(round(rmse,5)*1e5)}_it{i_iter}_ip{i_particle}_s{seed}.pkl",
                ),
                "wb",
            ) as f:
                dill.dump(m, f)
        except Exception as e:
            error_message = str(e)
            print(f"An error occurred: {error_message}")
            rmse = 99.99999  # error indicator

        rmse_list.append(rmse)
        # Clear model to reduce memory requirement
        m = None

    # Save the best model
    cost = min(rmse_list)
    seed = seeds[int(np.argmin(rmse_list))]
    with open(
        os.path.join(
            paths.exp_dir,
            f"{int(round(cost,5)*1e5)}_it{i_iter}_ip{i_particle}_s{seed}.txt",
        ),
        "w",
    ) as f:
        f.write(f"it{i_iter}_ip{i_particle}_s{seed}\nRMSE: {cost}\nx: {x}")

    return cost


# %% Setup PSO
# =============================================================================
# General settings
# =============================================================================
exp_dir = r"D:\SD6_exp_1f1w\sd6_cali"
os.chdir(exp_dir)

# Info
n_particles = 24
dimensions = 10
options = {
    "c1": 0.5,
    "c2": 0.5,
    "w": 0.8,
}  # hyperparameters {'c1', 'c2', 'w', 'k', 'p'}

# Bounds
lowerbounds = [141.1518, 60.152, -2.43, 3.5254, -0.9623, 0.8, 0.5, 0.5, 0, 0]  # [0]*4
upperbounds = [
    194.0593,
    69.4979,
    -1.9821,
    4.3674,
    -0.4535,
    1.2,
    1,
    1,
    0.5,
    0.5,
]  # [1]*4

rngen = np.random.default_rng(seed=12345)
init_pos = rngen.uniform(0, 1, (n_particles, dimensions))
for i in range(dimensions):
    init_pos[:, i] = init_pos[:, i] * (upperbounds[i] - lowerbounds[i]) + lowerbounds[i]
# %%
# Initialize PSO
optimizer = GlobalBestPSO(
    n_particles=n_particles,
    dimensions=dimensions,
    options=options,
    bounds=(lowerbounds, upperbounds),
    init_pos=init_pos,
    wd=exp_dir,
)

# N = 5
# rng = np.random.default_rng(12345)
# seeds = [int(rng.integers(low=0, high=999999)) for _ in range(N)]
seeds = [3, 56, 67]
# Run PSO
timer = TimeRecorder()
cost, pos = optimizer.optimize(
    obj_func, iters=100, n_processes=8, verbose=60, seeds=seeds, exp_dir=exp_dir
)

print("\a")
elapsed_time = timer.get_elapsed_time()
print(elapsed_time)

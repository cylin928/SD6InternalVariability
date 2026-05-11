
import os
import sys
import dill
import numpy as np
import pandas as pd
import pathnavigator
sys.setrecursionlimit(10000)  # Set to a higher value for dill deep dict.
root_dir = rf"/Users/{pathnavigator.user}/Documents/GitHub/SD6InternalVariability"
pn = pathnavigator.create(root_dir)
pn.code.chdir()
from py_champ.utility.util import TimeRecorder
from py_champ.models.particle_swarm import GlobalBestPSO
from py_champ.models.sd6_model_1f1w import SD6Model4SingleFieldAndWell
from utils import (
    load_x_from_cali_txt_output,
    cal_rmse_metrices_cali, 
    cal_weighted_rmse, 
    normalize_st_withdrawal, 
    update_model_inputs_from_cali_x
    )

# Set shortcut
pn.inputs.set_sc("input_pkl", "Inputs_SD6_2012_2022.pkl")
pn.data.set_sc("sd6_data", "Data_SD6_2012_2022.csv")

# Load inputs
with open(pn.sc.input_pkl, "rb") as f:
    inputs = dill.load(f)

# Load sd6 data
sd6_data = pd.read_csv(pn.sc.sd6_data, index_col=["year"]).loc[2012:2019, :]
sd6_data = normalize_st_withdrawal(sd6_data)

def run_simulation(x, seeds, inputs, sd6_data, exp_dir, **kwargs):
    import os
    import sys
    import numpy as np
    import pathnavigator
    sys.setrecursionlimit(10000)  # Set to a higher value for dill deep dict.
    root_dir = rf"/Users/{pathnavigator.user}/Documents/GitHub/SD6InternalVariability"
    pn = pathnavigator.create(root_dir)
    pn.code.chdir()
    pn.code.add_to_sys_path()
    from py_champ.models.sd6_model_1f1w import SD6Model4SingleFieldAndWell
    from utils import (
        cal_rmse_metrices_cali, 
        normalize_st_withdrawal, 
        update_model_inputs_from_cali_x,
        cal_weighted_rmse
        )
    # Prepare variables for simulation
    crop_options = ["corn", "others"]
    init_year = 2011
    #seed = 67

    (aquifers_dict, fields_dict, wells_dict, finances_dict, behaviors_dict, prec_aw_step, crop_price_step) = inputs
    fields_dict, crop_price_step, pars = update_model_inputs_from_cali_x(x, fields_dict, crop_price_step)
    try:
        rmse_list = []
        for seed in seeds:
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

            for i in range(8):
                m.step()
            m.end()
            df_sys, _ = m.get_dfs(m)
            
            # Normalize GW_st withdrawal to [0, 1] according to obv (i.e., data)
            df_sys = normalize_st_withdrawal(df_sys)
            
            # Calculate metrices
            metrices_cali = cal_rmse_metrices_cali(df_sys, sd6_data)
            
            # Calculate weighted RMSE
            rmse_weighted = cal_weighted_rmse(metrices_cali)["cali"]
            rmse_list.append(rmse_weighted)
            
            # Save memory
            m = None
        
        best_rmse = min(rmse_list)    
        mean_rmse = np.mean(rmse_list)
        median_rmse = np.median(rmse_list)
        
        # Save the best model based on median rmse
        ### Read PSO variables
        i_iter = kwargs.get("i_iter")
        i_particle = kwargs.get("i_particle")
        seed_best = seeds[int(np.argmin(rmse_list))] # not the median rmse, but the best rmse among seeds for this par set, to save the model.
        with open(
            os.path.join(
                exp_dir, "log",
                f"{int(round(median_rmse,5)*1e5)}_it{i_iter}_ip{i_particle}_s{seed_best}.txt",
            ),
            "w",
        ) as f:
            f.write(f"it{i_iter}_ip{i_particle}_s{seed_best}\nMedian RMSE: {median_rmse}\nMean RMSE: {mean_rmse}\nBest RMSE: {best_rmse}\nRMSE List: {rmse_list}\nx: {x}")
        return median_rmse
        
    except Exception as e:
        print(f"Error: {e}")
        return 100
#%% Run multiple seeds
# Setup PSO
# =============================================================================
# General settings
# =============================================================================
pn.outputs.mkdir("sd6_cali2")
pn.outputs.mkdir("sd6_cali2/log")
exp_dir = str(pn.outputs.get("sd6_cali2"))

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

rngen = np.random.default_rng(seed=39753) # 12345
init_pos = rngen.uniform(0, 1, (n_particles, dimensions))
for i in range(dimensions):
    init_pos[:, i] = init_pos[:, i] * (upperbounds[i] - lowerbounds[i]) + lowerbounds[i]
# %%
with open(pn.outputs.sd6_cali2.get("PSO_it70.pkl"), "rb") as f:
    load_dict = dill.load(f)

# Initialize PSO
optimizer = GlobalBestPSO(
    n_particles=n_particles,
    dimensions=dimensions,
    options=options,
    bounds=(lowerbounds, upperbounds),
    init_pos=init_pos,
    wd=exp_dir,
    load_dict=load_dict
)

# N = 5
# rng = np.random.default_rng(12345)
# seeds = [int(rng.integers(low=0, high=999999)) for _ in range(N)]
seeds = [3, 56, 67, 89, 123]
# Run PSO
timer = TimeRecorder()
cost, pos = optimizer.optimize(
    run_simulation, iters=100, n_processes=8, verbose=60, seeds=seeds, inputs=inputs, sd6_data=sd6_data, exp_dir=exp_dir
)

print("\a")
elapsed_time = timer.get_elapsed_time()
print(elapsed_time)
# %% Analysis
# with open(pn.outputs.sd6_cali2.get("PSO_it100.pkl"), "rb") as f:
#     load_dict = dill.load(f)

# sw = load_dict["swarm"]
# best_pos = sw.best_pos
# See run_multiple_seeds_on_a_given_par_set.py





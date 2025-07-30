import os
import sys
import dill
import time
from math import ceil
import numpy as np
import pandas as pd
from copy import deepcopy
from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler

set_loky_pickler("dill")
import pathnavigator

root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\SD6InternalVariability"
# root_dir = os.path.expanduser("~/Github/SD6InternalVariability")
pn = pathnavigator.create(root_dir)
pn.code.chdir()

from utils import load_x_from_cali_txt_output

# Define your sim function
def sim(x, inputs, scenario, df_init_crop, df_years, output_folder):
    """
    Run the model with the given scenario and parameters.

    Parameters
    ----------
    x : list
        A list of the calibrated parameters.
    inputs : tuple
        A tuple of input data.
    scenario : list
        The scenario of ratio values.
    df_init_crop : DataFrame
        The initial corn distribution dataframe.
    df_years : DataFrame
        The boostrapping years
    output_folder : Path
        The directory to save the experiment outputs.
    """
    import os
    import dill
    import time
    import numpy as np
    import pandas as pd
    from py_champ.models.sd6_model_1f1w import SD6Model4SingleFieldAndWell
    from py_champ.utility.util import dict_to_string

    (
        aquifers_dict,
        fields_dict,
        wells_dict,
        finances_dict,
        behaviors_dict,
        prec_aw_step_,
        crop_price_step_,
    ) = inputs

    # Scenario name
    scen_name = "-".join([str(x) for x in scenario])

    # Check if the file exists
    file_path = output_folder / f"df_sys-{scen_name}.csv"
    if file_path.exists():
        print(f"df_sys-{scen_name}.csv exist")
        return None
    file_path = output_folder / f"log-{scen_name}.txt"
    if file_path.exists():
        print(f"log-{scen_name}.txt exist")
        return None
    # Initiate the log
    log = {
        "scen_name": scen_name,
        "items": [
            "prec_ratio",
            "price_ratio",
            "corn_ratio",
            "corn_i",
            "boot_i",
            "seed",
        ],
        "scenario": scenario,
    }

    # Add the calibrated parameters to the model
    for fid in fields_dict:
        fields_dict[fid]["water_yield_curves"]["others"] = [
            x[0],
            x[1],
            x[2],
            x[3],
            x[4],
            0.1186,
        ]
    for yr in crop_price_step_["finance"]:
        crop_price_step_["finance"][yr]["others"] *= x[5]

    pars = {
        "perceived_risk": x[6],
        "forecast_trust": x[7],
        "sa_thre": x[8],
        "un_thre": x[9],
    }
    # ===================================================================================
    # scenario specific parameters
    # ["prec_ratio", "price_ratio", "corn_ratio", "corn_i", "boot_i", "seed"]

    year_seq = df_years[scenario[4]]
    init_year = 2011
    # Do the shaffling first (bootstrpping)
    prec_aw_step = {}
    for grid, precs in prec_aw_step_.items():
        prec_aw_step[grid] = {init_year + i: precs[yr] for i, yr in enumerate(year_seq)}

    crop_price_step = {}
    for fini, finance in crop_price_step_.items():
        crop_price_step[fini] = {
            init_year + i: finance[yr] for i, yr in enumerate(year_seq)
        }

    # Update prec_aw_step
    for grid, years in prec_aw_step.items():
        for year, crops in years.items():
            for crop, value in crops.items():
                if crop == "corn":
                    crops[crop] = min(
                        value * scenario[0],
                        fields_dict[fid]["water_yield_curves"]["corn"][1],
                    )
                elif crop == "others":
                    crops[crop] = min(
                        value * scenario[0],
                        fields_dict[fid]["water_yield_curves"]["others"][1],
                    )
    # Update crop_price_step
    for category, years in crop_price_step.items():
        for year, items in years.items():
            for item, value in items.items():
                items[item] = value * scenario[1]

    # Update corn_ratio (The input df_init_crop should already be associated with corn ratio)
    # Update initial crop according to the scenario
    for fid_i in range(336):
        fid = f"f{fid_i+1}"
        fields_dict[fid]["init"]["crop"] = df_init_crop.iloc[:, int(scenario[3])][fid_i]

    # ===================================================================================
    crop_options = ["corn", "others"]
    init_year = 2011
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
            seed=scenario[5],
        )

        for i in range(11):
            m.step()
        m.end()
    except Exception as e:
        print(f"Error in scenario {scenario}: {e}")
        log["error"] = str(e)
        log["success"] = "False"
        with open(output_folder / f"log-{scen_name}.txt", "w") as f:
            f.write(dict_to_string(log))
        return None

    # Output the results
    df_sys, df_agt = m.get_dfs(m)

    # Save model and dfs
    df_sys.to_csv(output_folder / f"df_sys-{scen_name}.csv", index=True)
    df_agt.to_csv(output_folder / f"df_agt-{scen_name}.csv", index=True)

    time.sleep(1)
    return None


# %%
# =======================================================================================
# Prepare the data and scenario list
# Load inputs
pn.outputs.mkdir("ANOVA")
pn.reload()
pn.inputs.set_sc("input_pkl", "Inputs_SD6_2012_2022.pkl")
pn.inputs.set_sc("scen_input_pkl", "Scenarios_SD6_2003_2022.pkl")
pn.models.set_sc("cali_x", "calibrated_parameters.txt")

with open(pn.sc.input_pkl, "rb") as f:
    inputs_ = dill.load(f)
with open(pn.sc.scen_input_pkl, "rb") as f:
    scen_inputs = dill.load(f)

inputs = (
    inputs_[0],
    inputs_[1],
    inputs_[2],
    inputs_[3],
    inputs_[4],
    scen_inputs[0],
    scen_inputs[1],
)

# Load calibrated parameters
cali_x = load_x_from_cali_txt_output(pn.sc.cali_x)

# Scenarios
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

df_init_crop_dict = {}
for ratio in ratio_dict["corn"]:
    df_init_crop = pd.read_csv(
        pn.scenarios.get(f"anova_C{ratio:.6f}_Rseed3256.csv")
    )
    df_init_crop[df_init_crop == 1] = "corn"
    df_init_crop[df_init_crop == 0] = "others"
    df_init_crop_dict[ratio] = df_init_crop


# Run simulations in parallel using joblib
output_folder = pn.outputs.ANOVA.get()

# Command-line input to determine which portion to run
if len(sys.argv) != 2:
    print("Usage: python exp4_anova.py <portion_index>")
    sys.exit(1)

nportions = 8
portion_index = int(sys.argv[1])  # Index of the portion to run (0 to 4)
if portion_index < 0 or portion_index >= nportions:
    print(f"Error: portion_index must be between 0 and {nportions-1}.")
    sys.exit(1)

# Split scenario_array into 5 portions
num_scenarios = len(scenario_array)
portion_size = ceil(num_scenarios / nportions)
start_index = portion_index * portion_size
end_index = min(start_index + portion_size, num_scenarios)
portion_scenario_array = scenario_array[start_index:end_index]

print(f"Start the simulations of portion {portion_index}")

Parallel(n_jobs=3)(
    delayed(sim)(
        x=cali_x,
        inputs=deepcopy(inputs),
        scenario=scenario,
        df_init_crop=df_init_crop_dict[scenario[2]],
        df_years=years,
        output_folder=output_folder,
    )
    for scenario in portion_scenario_array
)

print(f"All simulations of portion {portion_index} are completed")

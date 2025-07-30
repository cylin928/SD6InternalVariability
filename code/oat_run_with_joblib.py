import os
import dill
import time
import numpy as np
import pandas as pd
from copy import deepcopy
from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler

set_loky_pickler("dill")
import pathnavigator
from py_champ.models.sd6_model_1f1w import SD6Model4SingleFieldAndWell
from py_champ.utility.util import dict_to_string

root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\SD6InternalVariability"
pn = pathnavigator.create(root_dir)
pn.code.add_to_sys_path()

from utils import load_x_from_cali_txt_output


# Define your sim function
def sim(x, inputs, scenario, df_init_crop, output_folder, seed=67):
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
        prec_aw_step,
        crop_price_step,
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
        "items": ["prec", "crop_price", "elec_price", "B_mean", "aq_a", "corn_ratio"],
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
    for yr in crop_price_step["finance"]:
        crop_price_step["finance"][yr]["others"] *= x[5]

    pars = {
        "perceived_risk": x[6],
        "forecast_trust": x[7],
        "sa_thre": x[8],
        "un_thre": x[9],
    }
    # ===================================================================================
    # scenario specific parameters
    # [prec, crop_price, elec_price, B_mean, aq_a, corn_ratio]
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
    # Update elec_price
    finances_dict["finance"]["energy_price"] *= scenario[2]
    # Update B_mean
    for wid in wells_dict:
        wells_dict[wid]["init"]["B"] *= scenario[3]
    # Update aq_a
    for aq in aquifers_dict:
        aquifers_dict[aq]["aq_a"] *= scenario[4]
    # Update corn_ratio
    if scenario[5] != 1:
        df_init_crop[df_init_crop == 1] = "corn"
        df_init_crop[df_init_crop == 0] = "others"

        # Update initial crop according to the scenario
        for fid_i in range(336):
            fid = f"f{fid_i+1}"
            fields_dict[fid]["init"]["crop"] = df_init_crop[str(round(scenario[5], 1))][
                fid_i
            ]

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
            show_step=True,
            show_initialization=True,
            seed=seed,
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

    pre_dm_sols = m.behaviors["b171"].pre_dm_sols

    dm_sols = m.behaviors["b171"].dm_sols

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
seed = 2# 1, 2, 67
pn.outputs.mkdir(f"OAT_{seed}")
pn.inputs.set_sc("input_pkl", "Inputs_SD6_2012_2022.pkl")
pn.models.set_sc("cali_x", "calibrated_parameters.txt")

with open(pn.sc.input_pkl, "rb") as f:
    inputs = dill.load(f)

# Load calibrated parameters
cali_x = load_x_from_cali_txt_output(pn.sc.cali_x)

# ['prec', 'crop_price', 'elec_price', 'B_mean', 'aq_a', 'corn_ratio']
nvars = 6
ratio_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5]
scenario_array = np.ones((1 + len(ratio_list) * nvars, nvars))
for i in range(nvars):
    scenario_array[
        1 + i * len(ratio_list) : 1 + (i + 1) * len(ratio_list), i
    ] = ratio_list

# Run simulations in parallel using joblib
output_folder = pn.outputs.get(f"OAT_{seed}")
corn_csv_path = pn.scenarios.get() / "oat_corn_dist_Rseed3256.csv"
df_init_crop = pd.read_csv(corn_csv_path)

# scenario_array = [[1.4, 1, 1, 1, 1, 1]]
Parallel(n_jobs=4)(
    delayed(sim)(
        x=cali_x,
        inputs=deepcopy(inputs),
        scenario=scenario,
        df_init_crop=df_init_crop,
        output_folder=output_folder,
        seed=seed
    )
    for scenario in scenario_array
)

print("All simulations are completed")

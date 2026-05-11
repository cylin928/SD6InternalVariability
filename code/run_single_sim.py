import os
import sys
import dill
import numpy as np
import pandas as pd
import pathnavigator
from py_champ.models.sd6_model_1f1w import SD6Model4SingleFieldAndWell
from utils import load_x_from_cali_txt_output

sys.setrecursionlimit(10000)  # Set to a higher value for dill deep dict.
root_dir = rf"/Users/{pathnavigator.user}/Documents/GitHub/SD6InternalVariability"
pn = pathnavigator.create(root_dir)
pn.code.chdir()

# Try to load previous PSO
with open(pn.models.get("PSO_it99.pkl"), "rb") as f:
    pso = dill.load(f)

pso["swarm"].cost_history
swarm = pso["swarm"]
current_cost = swarm.current_cost


# Set shortcut
pn.inputs.set_sc("input_pkl", "Inputs_SD6_2012_2022.pkl")
pn.models.set_sc("cali_x", "calibrated_parameters.txt")
pn.data.set_sc("sd6_data", "Data_SD6_2012_2022.csv")

# Load inputs
with open(pn.sc.input_pkl, "rb") as f:
    inputs = dill.load(f)
(
    aquifers_dict,
    fields_dict,
    wells_dict,
    finances_dict,
    behaviors_dict,
    prec_aw_step,
    crop_price_step,
) = inputs

# Load calibrated parameters
x = load_x_from_cali_txt_output(pn.sc.cali_x)

# Load sd6 data
sd6_data = pd.read_csv(pn.sc.sd6_data, index_col=["year"])
# Normalize GW_st withdrawal to [0, 1] according to obv
sd6_data["GW_st"] = (sd6_data["GW_st"] - 17.5577) / (18.2131 - 17.5577)
sd6_data["withdrawal"] = (sd6_data["withdrawal"] - 1310.6749) / (
    3432.4528 - 1310.6749
)

# Prepare variables for simulation
crop_options = ["corn", "others"]
init_year = 2011
seed = 67
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

    df_sys, _ = m.get_dfs(m)
    
    # Normalize GW_st withdrawal to [0, 1] according to obv (i.e., data)
    df_sys = df_sys.loc[2012 :]
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


#%%


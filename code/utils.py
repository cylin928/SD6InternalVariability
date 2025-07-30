from dataclasses import dataclass, field
from typing import Dict, Union
import os
import warnings
import math
import re
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample


@dataclass
class ProjPaths:
    wd: str
    subfolders: Dict[str, str] = field(default_factory=dict)
    files: Dict[str, str] = field(default_factory=dict)
    otherpaths: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        # Change to working directory
        # os.chdir(self.wd)

        # Prepopulate with default directories
        self.add_subfolder("code")
        self.add_subfolder("data")
        self.add_subfolder("figures")
        self.add_subfolder("inputs")
        self.add_subfolder("outputs")
        self.add_subfolder("models")

        # Prepopulate with default files

    def check_name_eligibility(self, name: str) -> bool:
        """
        Checks if the name is eligible to be used as an attribute name.
        """
        is_eligible = name.isidentifier() and not hasattr(self, name)
        if is_eligible is False:
            raise ValueError(f"{name} has been used/is not a valid attribute name.")

    def check_path_existence_and_create(self, path: str) -> None:
        """
        Checks if the directory exists and creates it if not.
        """
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"New directory created at: {path}")

    def add_subfolder(self, subfolder: str) -> None:
        """
        Adds a subfolder path to the list of subfolders and sets it as an attribute.
        Ensures the subfolder path is relative to the working directory.

        Parameters
        ----------
        subfolder : str
            The name of the subfolder to add.
        """
        self.check_name_eligibility(subfolder)

        full_path = os.path.join(self.wd, subfolder)
        self.check_path_existence_and_create(full_path)
        self.subfolders[subfolder] = full_path
        setattr(self, subfolder, full_path)

    def remove_subfolder(self, subfolder: str) -> None:
        """
        Removes a subfolder path from the list of subfolders and deletes the attribute.

        Parameters
        ----------
        subfolder : str
            The name of the subfolder to remove.
        """
        if subfolder in self.subfolders:
            del self.subfolders[subfolder]
            if hasattr(self, subfolder):
                delattr(self, subfolder)
        else:
            warnings.warn(f"{subfolder} is not a tracked subfolder.", UserWarning)

    def add_nested_folder(self, parent_subfolder: str, nested_folder: str) -> None:
        """
        Adds a nested folder within a subfolder.

        Parameters
        ----------
        parent_subfolder : str
            The name of the parent subfolder.
        nested_folder : str
            The name of the nested folder to add.
        """
        self.check_name_eligibility(f"{parent_subfolder}_{nested_folder}")

        if parent_subfolder in self.subfolders:
            full_path = os.path.join(self.subfolders[parent_subfolder], nested_folder)
            self.check_path_existence_and_create(full_path)
            setattr(self, f"{parent_subfolder}_{nested_folder}", full_path)
        else:
            raise ValueError(f"{parent_subfolder} is not a tracked subfolder.")

    def remove_nested_folder(self, parent_subfolder: str, nested_folder: str) -> None:
        """
        Removes a nested folder within a subfolder.

        Parameters
        ----------
        parent_subfolder : str
            The name of the parent subfolder.
        nested_folder : str
            The name of the nested folder to remove.
        """
        nested_attr_name = f"{parent_subfolder}_{nested_folder}"
        if hasattr(self, nested_attr_name):
            full_path = getattr(self, nested_attr_name)
            if os.path.exists(full_path):
                os.rmdir(full_path)  # This only works if the directory is empty
            delattr(self, nested_attr_name)
        else:
            warnings.warn(
                f"{nested_attr_name} is not a tracked nested folder.", UserWarning
            )

    def add_file(self, file_name: str, subfolder: str = None, name: str = None) -> None:
        """
        Adds a file path to the list of files. If subfolder is specified, the file path is relative to that subfolder.
        Raises an error if the file does not exist.

        Parameters
        ----------
        file_name : str
            The name of the file to add.
        subfolder : str, optional
            The name of the subfolder to add the file to.
        """
        if subfolder:
            if subfolder in self.subfolders:
                full_path = os.path.join(self.subfolders[subfolder], file_name)
            else:
                raise ValueError(f"{subfolder} is not a tracked subfolder.")
        else:
            full_path = os.path.join(self.wd, file_name)

        if os.path.isfile(full_path):
            if name:
                self.check_name_eligibility(name)
                self.files[name] = full_path
                setattr(self, name, full_path)
            else:
                file_name_without_extension = os.path.splitext(file_name)[0]
                self.check_name_eligibility(file_name_without_extension)
                self.files[file_name_without_extension] = full_path
                setattr(self, file_name_without_extension, full_path)
        else:
            raise FileNotFoundError(f"{full_path} does not exist.")

    def remove_file(self, file_name: str) -> None:
        """
        Removes a file path from the list of files and deletes the attribute.

        Parameters
        ----------
        file_name : str
            The name of the file to remove.
        """
        if file_name in self.files:
            del self.files[file_name]
            if hasattr(self, file_name):
                delattr(self, file_name)
        else:
            warnings.warn(f"{file_name} is not a tracked file.", UserWarning)

    def add_other_path(self, name: str, path: str) -> None:
        """
        Adds a folder or file path to the dictionary of otherpaths and sets it as an
        attribute.

        Parameters
        ----------
        name : str
            The name of the attribute to add.
        path : str
            The path to add.
        """
        self.check_name_eligibility(name)
        is_dir = os.path.isdir(path)
        is_file = os.path.isfile(path)

        if is_dir:
            self.otherpaths[name] = path
            setattr(self, name, path)
        elif is_file:
            self.otherpaths[name] = path
            setattr(self, name, path)
        elif not is_dir and not is_file:
            if "." in path:  # Assume . means files
                raise FileNotFoundError(f"{path} does not exist.")
            else:
                try:
                    self.check_path_existence_and_create(path)
                    self.otherpaths[name] = path
                    setattr(self, name, path)
                except:
                    raise FileNotFoundError(f"{path} does not exist.")
        else:
            raise FileNotFoundError(f"{path} does not exist.")

    def remove_other_path(self, name: str) -> None:
        """
        Removes a folder or file path from the dictionary of otherpaths and deletes the
        attribute.

        Parameters
        ----------
        name : str
            The name of the attribute to remove.
        path : str
            The path to remove.
        """
        if name in self.otherpaths:
            del self.otherpaths[name]
            if hasattr(self, name):
                delattr(self, name)
        else:
            warnings.warn(f"{name} is not a tracked name of otherpaths.", UserWarning)

    def get_subfolders(self) -> Dict[str, str]:
        """
        Returns the dictionary of subfolder paths.
        """
        return self.subfolders

    def get_files(self) -> Dict[str, str]:
        """
        Returns the dictionary of file paths.
        """
        return self.files

    def get_otherpaths(self) -> Dict[str, str]:
        """
        Returns the dictionary of file paths.
        """
        return self.otherpaths


def get_section_index(section: str, num_of_scenarios: int):
    """
    Get the lower and upper index of the scenarios to be generated based on the section
    string.

    Parameters
    ----------
    section : str
        The section string specifying the range of scenarios to generate.
        E.g., "2-4/5" or "2/5".
    num_of_scenarios : int
        The total number of scenarios to generate.

    Returns
    -------
    lower_idx : int
        The lower index of the scenarios to generate.
    upper_idx : int
        The upper index of the scenarios to generate.
    """

    # section = "2-4/5" or "2/5"
    if "-" in section:
        block_from, block_to, n_parts = map(
            int, section.split("/")[0].split("-") + [section.split("/")[1]]
        )
    else:
        block, n_parts = map(int, section.split("/"))
        block_from = block
        block_to = block
    lower_idx = math.floor(num_of_scenarios / n_parts * (block_from - 1))
    upper_idx = math.ceil(num_of_scenarios / n_parts * block_to)
    return lower_idx, upper_idx


def load_x_from_cali_txt_output(file_path: str):
    """
    Load the calibrated parameters from the output file of the calibration process.

    Parameters
    ----------
    file_path : str
        The file path to the output file of the calibration process.

    Returns
    -------
    x_list : list
        The list of calibrated parameters
    """
    # Reading the content of the file
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Finding the line index where 'x:' starts
    x_start_index = next(i for i, line in enumerate(lines) if line.startswith("x:"))

    # Concatenating all lines following the 'x:' line
    x_lines = (
        "".join(lines[x_start_index:])
        .replace("x:", "")
        .replace("[", "")
        .replace("]", "")
        .strip()
    )

    # Extracting the numerical values from the concatenated 'x' lines
    x_values = re.findall(r"[-+]?\d*\.\d+e[+-]\d+|[-+]?\d*\.\d+", x_lines)

    # Converting the string values to floats
    x_list = [float(value) for value in x_values]

    # Formatting the output
    param_info = lines[0].strip()
    rmse_info = lines[1].strip()
    print(f"Load calibrated parameters from {param_info} ({rmse_info}):")
    print(x_list)
    return x_list


def bootstrap(data: list, num_samples: int, replace: bool = True, seed: int = None):
    """
    Perform bootstrapping on a given list.

    Parameters:
    ----------
    data : list
        List of data to be bootstrapped.
    num_samples : int
        Number of bootstrap samples to generate.
    replace : bool
        Whether the sample is with or without replacement. Default is True, meaning that
        a value of a can be selected multiple times.
        If false, permutation resampling is performed instead of bootstrapping.
    """
    data = np.array(data)
    n = len(data)

    rng = np.random.default_rng(seed)

    bootstrap_samples = []
    for _ in range(num_samples):
        bootstrap_sample = rng.choice(data, size=n, replace=replace).tolist()
        bootstrap_samples.append(bootstrap_sample)

    return bootstrap_samples


def convert_df_sys_units(df_sys: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the units of the df_sys results.
    Withdrawal: 10^4 to 10^6 m3
    CONSUMAT states: from number of agents to ratios

    Parameters
    ----------
    df_sys : pd.DataFrame
        The df_sys results of the sd6_model or a dataframe that has columns, withdrawal,
        'Imitation', 'Social comparison', 'Repetition', 'Deliberation'.

    Returns
    -------
    df_sys : pd.DataFrame
        The df_sys results with converted units.
    """
    df_sys["withdrawal"] /= 100  # 10^4 to 10^6 m3
    states = ["Imitation", "Social comparison", "Repetition", "Deliberation"]
    try:
        total_agt_number = df_sys[states].sum(axis=1)
        for state in states:
            df_sys[state] /= total_agt_number
    except KeyError:
        print("No CONSUMAT states in the dataframe to convert.")
        pass
    return df_sys


def read_scenario_results(
    scenario_name: str,
    prefix: str,
    result_dir: str,
    file_type: str = "df_sys",
    ext: str = ".csv",
    convert_units: bool = True,
) -> Union[pd.DataFrame, object]:
    """
    Read the results of a scenario from the result directory.

    Parameters
    ----------
    scenario_name : str
        The name of the scenario.
    prefix : str
        The prefix of the scenario.
    result_dir : str
        The directory where the results are stored.
    file_type : str
        The type of the result file to read. Default is 'df_sys'.
    ext : str
        The extension of the result file. Default is '.csv'.

    Returns
    -------
    df : pd.DataFrame or objects
        The results of the scenario.
    """
    filename = f"{file_type}-{prefix}-{scenario_name}{ext}"
    df = pd.read_csv(os.path.join(result_dir, filename))
    df["scenario_name"] = scenario_name
    df.set_index("scenario_name", inplace=True)
    if convert_units:
        if file_type == "df_sys":
            df = convert_df_sys_units(df)
    return df


def compute_metrices_for_df_sys(df_sys: str, scenario_name: str) -> pd.DataFrame:
    """
    Compute the metrices for the df_sys results of a scenario.

    Parameters
    ----------
    df_sys : pd.DataFrame
        The df_sys results of a scenario.
    scenario_name : str
        The name of the scenario.

    Returns
    -------
    df_long : pd.DataFrame
        The metrices of the scenario in a long format.
    """
    periods = {
        "LEMA": [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
        "LEMA1": [2013, 2014, 2015, 2016, 2017],
        "LEMA2": [2018, 2019, 2020, 2021, 2022],
        "Dry": [2013, 2020, 2022],
        "Wet": [2017, 2018, 2019],
    }

    if df_sys.index.name != "year":
        df_sys.reset_index(inplace=True)
        df_sys.set_index("year", inplace=True)

    df = pd.DataFrame()
    for period, years in periods.items():
        df[period] = df_sys.loc[years].mean(axis=0, numeric_only=True)

    # reverse pivot table
    df_long = df.reset_index().melt(
        id_vars=["index"], var_name="period", value_name="value"
    )
    df_long.rename(columns={"index": "variable"}, inplace=True)
    # add scenario name
    df_long["scenario_name"] = scenario_name
    df_long.set_index("scenario_name", inplace=True)

    return df_long


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


def form_df_sys_ts(
    prefix: str,
    scenario_index: pd.DataFrame | list,
    result_dir: str | list,
    output_dir: str | list,
    overwrite: bool = False,
    cal_state_change=True,
):
    """
    Form a long time series dataframe from the df_sys results of the scenarios.

    Parameters
    ----------
    prefix : str
        The prefix of the scenarios.
    scenario_index : pd.DataFrame
        The index of the scenarios.
    result_dir : str
        The directory where the results are stored.
    output_dir : str
        The directory where the output file is stored.
    overwrite : bool
        Whether to overwrite the existing file. Default is False.

    Returns
    -------
    df_ts : pd.DataFrame
        The long time series dataframe of the scenarios.
    """
    if isinstance(scenario_index, pd.DataFrame):
        output_file_path = os.path.join(output_dir, f"{prefix}_df_sys_ts.csv")
        if os.path.exists(output_file_path) and not overwrite:
            warnings.warn(
                f"Warning: The file '{output_file_path}' already exists.\n"
                + "The existed file is loaded without reprocessing.",
                UserWarning,
            )
            return pd.read_csv(output_file_path, index_col=[0])

        # Load the results to create a long ts df.
        df_sys_ts = []
        for scenario_name in tqdm(scenario_index.index):
            df_sys = read_scenario_results(scenario_name, prefix, result_dir)
            if cal_state_change:
                df_agt = read_scenario_results(
                    scenario_name, prefix, result_dir, "df_agt"
                )
                df_sys["state_change"] = count_state_changes_per_year(df_agt)
            df_sys_ts.append(df_sys)
        df_sys_ts = pd.concat(df_sys_ts)

        df_ts = pd.merge(
            df_sys_ts, scenario_index, how="left", left_index=True, right_index=True
        )
    else:
        df_ts_list = []
        for i in range(len(scenario_index)):
            output_file_path = os.path.join(output_dir[i], f"{prefix}_df_sys_ts.csv")
            # if os.path.exists(output_file_path) and not overwrite:
            #     warnings.warn(
            #         f"Warning: The file '{output_file_path}' already exists.\n"
            #         + "The existed file is loaded without reprocessing.",
            #         UserWarning,
            #     )
            #     return pd.read_csv(output_file_path, index_col=[0])

            df_sys_ts = []
            for scenario_name in tqdm(scenario_index[i].index):
                df_sys = read_scenario_results(
                    scenario_name, prefix, result_dir[i], convert_units=False
                )
                if cal_state_change:
                    df_agt = read_scenario_results(
                        scenario_name, prefix, result_dir[i], "df_agt"
                    )
                    df_sys["state_change"] = count_state_changes_per_year(df_agt)
                df_sys_ts.append(df_sys)
            df_sys_ts = pd.concat(df_sys_ts)
            df_ts = pd.merge(
                df_sys_ts,
                scenario_index[i],
                how="left",
                left_index=True,
                right_index=True,
            )
            df_ts_list.append(df_ts)
        df_ts = pd.concat(df_ts_list)
    df_ts.to_csv(output_file_path)
    return df_ts


def form_df_sys_metrices(
    prefix: str,
    scenario_index: pd.DataFrame | list,
    result_dir: str | list,
    output_dir: str,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Form a long metrices dataframe from the df_sys results of the scenarios.

    Parameters
    ----------
    prefix : str
        The prefix of the scenarios.
    scenario_index : pd.DataFrame
        The index of the scenarios.
    result_dir : str
        The directory where the results are stored.
    output_dir : str
        The directory where the output file is stored.
    overwrite : bool
        Whether to overwrite the existing file. Default is False.

    Returns
    -------
    df_metrices : pd.DataFrame
        The long metrices dataframe of the scenarios.
    """
    output_file_path = os.path.join(output_dir, f"{prefix}_df_sys_metrices.csv")
    if os.path.exists(output_file_path) and not overwrite:
        warnings.warn(
            f"Warning: The file '{output_file_path}' already exists.\n"
            + "The existed file is loaded without reprocessing.",
            UserWarning,
        )
        return pd.read_csv(output_file_path, index_col=[0])

    if isinstance(scenario_index, pd.DataFrame):
        # Load the results to create a long metrices df.
        df_sys_metrices = pd.concat(
            [
                compute_metrices_for_df_sys(
                    df_sys=read_scenario_results(scenario_name, prefix, result_dir),
                    scenario_name=scenario_name,
                )
                for scenario_name in tqdm(scenario_index.index)
            ]
        )

        df_metrices = pd.merge(
            df_sys_metrices,
            scenario_index,
            how="left",
            left_index=True,
            right_index=True,
        )
    else:
        df_metrices_list = []
        for i in range(len(scenario_index)):
            # Load the results to create a long metrices df.
            df_sys_metrices = pd.concat(
                [
                    compute_metrices_for_df_sys(
                        df_sys=read_scenario_results(
                            scenario_name, prefix, result_dir[i]
                        ),
                        scenario_name=scenario_name,
                    )
                    for scenario_name in tqdm(scenario_index[i].index)
                ]
            )

            df_metrices = pd.merge(
                df_sys_metrices,
                scenario_index[i],
                how="left",
                left_index=True,
                right_index=True,
            )
            df_metrices_list.append(df_metrices)
        df_metrices = pd.concat(df_metrices_list)
    df_metrices.to_csv(output_file_path)
    return df_metrices


def reassign_corn_ratio_groups(
    df: pd.DataFrame, from_interval: float = 0.01, to_interval: float = 0.05
):
    """
    Reassign the corn ratio groups based on the new interval.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the corn ratio groups.
    from_interval : float
        The interval of the corn ratio groups in the original dataframe.
    to_interval : float
        The interval of the corn ratio groups in the new dataframe.

    Returns
    -------
    df : pd.DataFrame
        The dataframe with the reassigned corn ratio groups.
    """
    sub_interval = int(from_interval * 100)
    interval = int(to_interval * 100)
    interval_list = np.arange(1, 100 + interval, interval)
    for i in range(len(interval_list) - 1):
        selected_sub_intervals = np.arange(
            interval_list[i], interval_list[i + 1], sub_interval
        )
        group_name = f"CG{np.mean(selected_sub_intervals).astype(int):02d}"
        names_to_select = [f"C{i:02d}" for i in selected_sub_intervals]
        df.loc[df["ratio"].isin(names_to_select), "ratio_group"] = group_name
    return df


def count_duplicate_rows(arr: np.ndarray, show: bool = True) -> bool:
    """
    Check if an array has any duplicate rows.

    Parameters
    ----------
    arr : np.ndarray
        The input array.
    show : bool
        Whether to show the counts of unique rows. Default is True.

    Returns
    -------
    bool
        True if there are any duplicate rows, False otherwise.
    """
    # Convert to a structured array view
    structured_arr = np.ascontiguousarray(arr).view(
        np.dtype((np.void, arr.dtype.itemsize * arr.shape[1]))
    )

    # Find unique rows and their counts
    _, unique_counts = np.unique(structured_arr, return_counts=True)
    if show:
        print(unique_counts)
    # Check if any row count is greater than 1
    return np.any(unique_counts > 1)


def form_df_Y(
    scenario_index: pd.DataFrame,
    df_metrices: pd.DataFrame,
    df_ts: pd.DataFrame,
    problem,
    N: int = 700,
    py_seed: int | None = None,
    count_duplicate_rows: bool = False,
):
    # Define the problem
    sobol_seed = 6435

    # Generate Sobol samples
    param_values = sobol_sample.sample(
        problem, N=N, calc_second_order=True, seed=sobol_seed
    )
    # Convert to integer indices that will be used to map to the realizations in
    # scenario_index.
    param_values = param_values.astype(int)
    # Just to check if there are any duplicate rows in the param_values.
    if count_duplicate_rows:
        count_duplicate_rows(param_values)

    # Map the indices to the realizations in scenario_index
    mapping_dict = {
        var_: sorted(scenario_index[var_].unique()) for var_ in problem["names"]
    }
    df_Y = pd.DataFrame(param_values)
    df_Y.columns = problem["names"]

    # Function to combine columns
    def combine_columns_py_seed(row, py_seed: int = None):
        set_ = mapping_dict["set"][row["set"]]
        prec_id = mapping_dict["prec_id"][row["prec_id"]]
        eco_id = mapping_dict["eco_id"][row["eco_id"]]
        if py_seed is None:
            py_seed = mapping_dict["py_seed"][row["py_seed"]]
        return f"R00_C00_{set_}_{prec_id}_{eco_id}_Pseed{py_seed}"

    # Apply the function to create a new column
    df_Y.index = df_Y.apply(combine_columns_py_seed, axis=1, py_seed=py_seed).to_list()
    # Combine all variables
    df_metrices["var_period"] = (
        df_metrices["variable"] + " (" + df_metrices["period"] + ")"
    )
    df = df_metrices[["var_period", "value"]]
    dff = df_metrices.pivot_table(
        values="value", index="scenario_name", columns="var_period"
    )
    df_Y = pd.merge(df_Y, dff, how="left", left_index=True, right_index=True)
    var_list = [
        "GW_st",
        "withdrawal",
        "rainfed",
        "corn",
        "others",
        "Imitation",
        "Social comparison",
        "Repetition",
        "Deliberation",
    ]
    for year in range(2012, 2023):
        df = df_ts.loc[df_ts["year"] == year, var_list]
        df.columns = [f"{v} ({year})" for v in var_list]
        df_Y = pd.merge(df_Y, df, how="left", left_index=True, right_index=True)
    return df_Y


def cal_sobol_indices(
    var_list: list,
    df_Y: pd.DataFrame,
    problem: dict,
    zero_thres: float = 0,
    par_name_list: list | None = None,
) -> tuple:
    """
    Calculate the Sobol indices for the given variables.

    Parameters
    ----------
    var_list : list
        The list of variables to calculate the Sobol indices.
    df_Y : pd.DataFrame
        The dataframe of the output variables.
    problem : dict
        The problem dictionary.
    zero_thres : float
        The threshold to consider the value as zero. Default is 0.
    par_name_list : list, optional
        The list of parameter names that will be used to overwrite the names defined in
        problem. Default is None.

    Returns
    -------
    df_s12 : pd.DataFrame
        The dataframe of the first and second order Sobol indices.
    df_st : pd.DataFrame
        The dataframe of the total Sobol indices.
    """
    df_s12 = pd.DataFrame()
    df_st = pd.DataFrame()
    for var_ in var_list:
        Y = df_Y[var_].values
        Si = sobol_analyze.analyze(problem, Y)
        si = list(Si["S1"])
        s2 = Si["S2"]
        for i in range(1, s2.shape[0]):
            si += list(s2[i - 1, i:])
        df_s12[var_] = si
        df_st[var_] = Si["ST"]

    if par_name_list is None:
        par_name_list = problem["names"]
    names = par_name_list.copy()
    for i in range(0, len(par_name_list) - 1):
        for j in range(i + 1, len(par_name_list)):
            names.append(f"{par_name_list[i]}-{par_name_list[j]}")

    df_s12.index = names
    df_st.index = par_name_list

    df_s12[(df_s12 < 0) & (df_s12 >= zero_thres)] = 0
    df_st[(df_st < 0) & (df_st >= zero_thres)] = 0
    return df_s12.T, df_st.T

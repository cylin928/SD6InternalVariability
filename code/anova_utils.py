import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import seaborn as sns
import clt

var_dict = {
    'ST': 'Saturated\nthickness',
    'Wi': 'Withdrawal',
    'RF': 'Rainfed\nfield ratio',
    'CF': 'Field ratio\nfor corn',
    'OF': 'Field ratio\nfor others',
    'CSC': 'Behavioral\nstate changes',
    'TP': 'Total\nprofit'
    }

def anova_yr(y_out, df_sys_all, yr, seed=None, plot_residual=False):
    """
    Conduct ANOVA for a specific year and optionally plot residuals.

    Parameters
    ==========
    y_out : str
        The dependent variable for the ANOVA.
    df_sys_all : DataFrame
        The DataFrame containing the data for the ANOVA.
    yr : int
        The year for which to conduct the ANOVA.
    seed : int, optional
        The seed for the simulation. If None, all seeds are included.
    """
    df = df_sys_all[df_sys_all["Year"] == yr]

    if seed is not None:
        df = df[df["Seed"] == seed]
    formula = f"{y_out} ~ Pr + Cr + Co + Pr:Cr + Pr:Co + Cr:Co + Pr:Cr:Co"
    model = ols(formula, data=df).fit()
    print(model.summary())

    anova_results = anova_lm(model, typ=2)
    print("ANOVA Results:\n", anova_results)

    if plot_residual:
        # This is a histogram of the residuals of the model to check the normality assumption.
        residuals = model.resid
        fig, ax = plt.subplots(figsize=(4, 5))
        sns.histplot(residuals, kde=True, bins=50, color='skyblue', ax=ax)
        ax.set_title(f"Histogram of Model Residuals in Year {yr}")
        ax.set_xlabel("Residual")
        ax.set_ylabel("Frequency")
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    return anova_results

def get_sum_sq_over_years(y_out, df_sys_all, seed=None, to_fraction=True):
    """
    Get normalized sum of squares over years for a specific seed.

    Parameters
    ==========
    y_out : str
        The dependent variable for the ANOVA.
    df_sys_all : DataFrame
        The DataFrame containing the data for the ANOVA.
    seed : int, optional
        The seed for the simulation. If None, all seeds are included.
    to_fraction : bool, optional
        Whether to normalize the sum of squares.
    """
    sum_sq = pd.DataFrame()
    for yr in range(2013, 2023):
        sum_sq[yr] = anova_yr(y_out, df_sys_all, yr, seed)["sum_sq"]
    sum_sq = sum_sq.T
    if to_fraction: # Sum of all sum of squares = 1
        sum_sq = sum_sq.div(sum_sq.sum(axis=1), axis=0)
    sum_sq["Seed"] = seed
    return sum_sq

def get_mu_sd_dfs_over_seeds(y_out, df_sys_all, aggre_interaction_term=True, to_fraction=True):
    """
    Get mean and standard deviation of normalized sum of squares over seeds.

    Parameters
    ==========
    y_out : str
        The dependent variable for the ANOVA.
    df_sys_all : DataFrame
        The DataFrame containing the data for the ANOVA.
    aggre_interaction_term : bool, optional
        Whether to aggregate interaction terms into a single column.
    """
    sum_sq_nor = []
    for seed in [1, 2, 67]: #[67]:#
        df = get_sum_sq_over_years(y_out, df_sys_all, seed=seed, to_fraction=to_fraction)
        if aggre_interaction_term:
            cols = [i for i in df.columns if ":" in i]
            df["Interaction terms"] = df[cols].sum(axis=1)
            df = df.drop(cols, axis=1)
        sum_sq_nor.append(df)

    # Mean over seeds
    sum_sq_nor_mu = pd.concat(sum_sq_nor).groupby(level=0).mean()

    # Calculate cumulative sum for each row for stacked bar plot
    sum_sq_nor_cumsum = []
    for df in sum_sq_nor:
        df.loc[:, df.columns[:-1]] = df.loc[:, df.columns[:-1]].cumsum(axis=1)
        sum_sq_nor_cumsum.append(df)

    # Standard deviation over seeds
    sum_sq_nor_std = pd.concat(sum_sq_nor_cumsum).groupby(level=0).std()
    return sum_sq_nor_mu.drop("Seed", axis=1), sum_sq_nor_std.drop("Seed", axis=1)

def plot_anova_sum_sq_fraction(mu_dict, save_figname=None):
    vlist = ['ST', 'CF', 'Wi', 'CSC']
    df_mean = []
    for v in vlist:
        df_mean.append(mu_dict[v].mean().to_frame(v))
    df_mean = pd.concat(df_mean, axis=1).T

    fig = plt.figure(figsize=(6.5, 4))
    gs = GridSpec(
        2, 5,
        width_ratios=[4, 0.6, 1.8, 4, 0.6],  # slightly shrink mean panels
        height_ratios=[1, 1],
        wspace=0.1,  # space between columns
        hspace=0.28,  # space between rows
        figure=fig
    )

    # clt.open_cmap_manual()
    cm = clt.cmap.Colormap('petroff:petroff10')
    # new_order = [0, 1, 5, 3, 4]  # Example: Custom order
    # reordered_colors = [cm(i) for i in new_order]
    #                   Pr, Cr, Co, IV, Error, interaction
    reordered_colors = [cm(0), cm(1), cm(5), cm(3), "black", cm(4)]
    vars_order = ["Pr", "Cr", "Co", "IV", "Error", "Interaction terms"]
    def add_anova(ax, v, show_xticks=True):
        name = var_dict[v]
        mu = mu_dict[v][vars_order]
        mu.plot(kind='bar', stacked=True, ax=ax, width=0.7, legend=False, color=reordered_colors, edgecolor='dimgray')
        ax.set_ylabel(name, fontsize=10)
        ax.set_ylim([0,1])
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        if show_xticks is False:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Year", fontsize=10)

    def add_anova_mean(ax, v, show_xticks=True):
        mu = df_mean.loc[[v], vars_order]
        mu.index = ["Mean"]
        mu.plot(kind='bar', stacked=True, ax=ax, width=0.7, legend=False, color=reordered_colors, edgecolor='dimgray')
        #ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_ylim([0,1])
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        if show_xticks is False:
            ax.set_xticklabels([])
        # Remove all four spines (top, right, left, bottom)
        #for spine in ['top', 'right', 'left', 'bottom']:
        #    ax.spines[spine].set_visible(False)

    # Add ANOVA plots
    v = 'Wi'
    ax = fig.add_subplot(gs[0, 0])
    add_anova(ax, v, show_xticks=False)
    # Create a single legend for both subplots
    handles, labels = ax.get_legend_handles_labels()
    map_dict = {
        "Pr": "Prec ratio\n(Pr)",
        "Cr": "Crop price\nratio (Cr)",
        "Co": "Corn field\nratio (Co)",
        "IV": "Internal\nvariability",
        "Error": "Model\nerror",
        "Interaction terms": "Interaction\nterms"}
    labels = [map_dict[i] if map_dict.get(i) is not None else i for i in labels]
    fig.legend(handles, labels, ncols=len(vars_order), bbox_to_anchor=(0.48, -0.05), loc='upper center', frameon=False, fontsize=8)
    v = 'ST'
    ax = fig.add_subplot(gs[1, 0])
    add_anova(ax, v)
    v = 'CF'
    ax = fig.add_subplot(gs[0, 3])
    add_anova(ax, v, show_xticks=False)
    v = 'CSC'
    ax = fig.add_subplot(gs[1, 3])
    add_anova(ax, v)

    # Add mean plots
    v = 'Wi'
    ax = fig.add_subplot(gs[0, 1])
    add_anova_mean(ax, v, show_xticks=False)
    v = 'ST'
    ax = fig.add_subplot(gs[1, 1])
    add_anova_mean(ax, v)
    v = 'CF'
    ax = fig.add_subplot(gs[0, 4])
    add_anova_mean(ax, v, show_xticks=False)
    v = 'CSC'
    ax = fig.add_subplot(gs[1, 4])
    add_anova_mean(ax, v)

    # Add section titles (in place of axes[0/1].set_title)
    fig.text(0.11, 0.99, "Policy-relevant indicator", fontsize=12, va='top', ha='left')
    fig.text(0.534, 0.99, "Human behavioral variable", fontsize=12, va='top', ha='left')
    fig.text(-0.02, 0.5, "Fraction of variance explained", va='center', ha='center',
             rotation='vertical', fontsize=12)

    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    y1 = 0.93; y2 = 0.497
    positions = [
        (0.12, y1),  # for top-left subplot
        (0.404, y1),  # for top-right subplot
        (0.12, y2),  # for bottom-left subplot
        (0.404, y2),  # for bottom-right subplot
        (0.58, y1),  # for top-left subplot
        (0.862, y1),  # for top-right subplot
        (0.58, y2),  # for bottom-left subplot
        (0.862, y2),  # for bottom-right subplot
    ]

    for label, (x, y) in zip(labels, positions):
        fig.text(x, y, label, fontsize=10, fontweight='bold', ha='left', va='top')

    # Show the plot
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    if save_figname is not None:
        fig.savefig(save_figname, dpi=300, bbox_inches='tight')
    plt.show()

def plot_anova_sum_sq_fraction_old(mu_dict, save_figname=None):
    vlist = ['ST', 'CF', 'Wi', 'CSC']
    df_mean = []
    for v in vlist:
        df_mean.append(mu_dict[v].mean().to_frame(v))
    df_mean = pd.concat(df_mean, axis=1).T

    fig = plt.figure(figsize=(6.5, 4))
    gs = GridSpec(
        2, 5,
        width_ratios=[4, 0.6, 1.8, 4, 0.6],  # slightly shrink mean panels
        height_ratios=[1, 1],
        wspace=0.1,  # space between columns
        hspace=0.28,  # space between rows
        figure=fig
    )

    # clt.open_cmap_manual()
    cm = clt.cmap.Colormap('petroff:petroff10')
    new_order = [0, 1, 5, 3, 4]  # Example: Custom order
    reordered_colors = [cm(i) for i in new_order]

    def add_anova(ax, v, show_xticks=True):
        name = var_dict[v]
        mu = mu_dict[v]
        mu.plot(kind='bar', stacked=True, ax=ax, width=0.7, legend=False, color=reordered_colors, edgecolor='dimgray')
        ax.set_ylabel(name, fontsize=10)
        ax.set_ylim([0,1])
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        if show_xticks is False:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Year", fontsize=10)

    def add_anova_mean(ax, v, show_xticks=True):
        mu = df_mean.loc[[v], :]
        mu.index = ["Mean"]
        mu.plot(kind='bar', stacked=True, ax=ax, width=0.7, legend=False, color=reordered_colors, edgecolor='dimgray')
        #ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_ylim([0,1])
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        if show_xticks is False:
            ax.set_xticklabels([])
        # Remove all four spines (top, right, left, bottom)
        #for spine in ['top', 'right', 'left', 'bottom']:
        #    ax.spines[spine].set_visible(False)

    # Add ANOVA plots
    v = 'Wi'
    ax = fig.add_subplot(gs[0, 0])
    add_anova(ax, v, show_xticks=False)
    # Create a single legend for both subplots
    handles, labels = ax.get_legend_handles_labels()
    map_dict = {
        "Pr": "Prec ratio\n(Pr)",
        "Cr": "Crop price\nratio (Cr)",
        "Co": "Corn field\nratio (Co)",
        "Residual": "Internal\nvariability",
        "Interaction terms": "Interaction\nterms"}
    labels = [map_dict[i] if map_dict.get(i) is not None else i for i in labels]
    fig.legend(handles, labels, ncols=5, bbox_to_anchor=(0.48, -0.05), loc='upper center', frameon=False, fontsize=8)
    v = 'ST'
    ax = fig.add_subplot(gs[1, 0])
    add_anova(ax, v)
    v = 'CF'
    ax = fig.add_subplot(gs[0, 3])
    add_anova(ax, v, show_xticks=False)
    v = 'CSC'
    ax = fig.add_subplot(gs[1, 3])
    add_anova(ax, v)

    # Add mean plots
    v = 'Wi'
    ax = fig.add_subplot(gs[0, 1])
    add_anova_mean(ax, v, show_xticks=False)
    v = 'ST'
    ax = fig.add_subplot(gs[1, 1])
    add_anova_mean(ax, v)
    v = 'CF'
    ax = fig.add_subplot(gs[0, 4])
    add_anova_mean(ax, v, show_xticks=False)
    v = 'CSC'
    ax = fig.add_subplot(gs[1, 4])
    add_anova_mean(ax, v)

    # Add section titles (in place of axes[0/1].set_title)
    fig.text(0.11, 0.99, "Policy-relevant indicator", fontsize=12, va='top', ha='left')
    fig.text(0.534, 0.99, "Human behavioral variable", fontsize=12, va='top', ha='left')
    fig.text(-0.02, 0.5, "Fraction of variance explained", va='center', ha='center',
             rotation='vertical', fontsize=12)

    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    y1 = 0.93; y2 = 0.497
    positions = [
        (0.12, y1),  # for top-left subplot
        (0.404, y1),  # for top-right subplot
        (0.12, y2),  # for bottom-left subplot
        (0.404, y2),  # for bottom-right subplot
        (0.58, y1),  # for top-left subplot
        (0.862, y1),  # for top-right subplot
        (0.58, y2),  # for bottom-left subplot
        (0.862, y2),  # for bottom-right subplot
    ]

    for label, (x, y) in zip(labels, positions):
        fig.text(x, y, label, fontsize=10, fontweight='bold', ha='left', va='top')

    # Show the plot
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    if save_figname is not None:
        fig.savefig(save_figname, dpi=300, bbox_inches='tight')
    plt.show()

def plot_anova_sum_sq(mu_dict, save_figname=None):
    vlist = ['ST', 'CF', 'Wi', 'CSC']
    df_mean = []
    for v in vlist:
        df_mean.append(mu_dict[v].mean().to_frame(v))
    df_mean = pd.concat(df_mean, axis=1).T

    fig = plt.figure(figsize=(6.5, 4))
    gs = GridSpec(
        2, 5,
        width_ratios=[4, 0.6, 1.8, 4, 0.6],  # slightly shrink mean panels
        height_ratios=[1, 1],
        wspace=0.1,  # space between columns
        hspace=0.28,  # space between rows
        figure=fig
    )

    # clt.open_cmap_manual()
    cm = clt.cmap.Colormap('petroff:petroff10')
    new_order = [0, 1, 5, 3, 4]  # Example: Custom order
    reordered_colors = [cm(i) for i in new_order]

    def add_anova(ax, v, show_xticks=True):
        name = var_dict[v]
        mu = mu_dict[v]
        mu.plot(kind='bar', stacked=True, ax=ax, width=0.7, legend=False, color=reordered_colors, edgecolor='dimgray')

        ax.grid(True, axis='y', linestyle='--', alpha=0.5)

        ax.set_ylabel(name, fontsize=10)
        ax.set_ylim(bottom=0)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.yaxis.get_major_formatter().set_useOffset(True)
        ax.yaxis.get_major_formatter().set_scientific(True)
        ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)

        if show_xticks is False:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Year", fontsize=10)

    def add_anova_mean(ax, v, ylim, yticks, show_xticks=True):
        mu = df_mean.loc[[v], :]
        mu.index = ["Mean"]
        mu.plot(kind='bar', stacked=True, ax=ax, width=0.7, legend=False, color=reordered_colors, edgecolor='dimgray')
        ax.set_yticks(yticks)
        ax.set_yticklabels([])
        ax.set_ylim(ylim)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        if show_xticks is False:
            ax.set_xticklabels([])
        # Remove all four spines (top, right, left, bottom)
        # for spine in ['top', 'right', 'left', 'bottom']:
        #     ax.spines[spine].set_visible(False)

    # Add ANOVA plots
    v = 'Wi'
    ax1 = fig.add_subplot(gs[0, 0])
    add_anova(ax1, v, show_xticks=False)
    # Create a single legend for both subplots
    handles, labels = ax1.get_legend_handles_labels()
    map_dict = {
        "Pr": "Prec ratio\n(Pr)",
        "Cr": "Crop price\nratio (Cr)",
        "Co": "Corn field\nratio (Co)",
        "Residual": "Internal\nvariability",
        "Interaction terms": "Interaction\nterms"}
    labels = [map_dict[i] if map_dict.get(i) is not None else i for i in labels]
    fig.legend(handles, labels, ncols=5, bbox_to_anchor=(0.48, -0.05), loc='upper center', frameon=False, fontsize=8)
    v = 'ST'
    ax2 = fig.add_subplot(gs[1, 0])
    add_anova(ax2, v)
    v = 'CF'
    ax3 = fig.add_subplot(gs[0, 3])
    add_anova(ax3, v, show_xticks=False)
    v = 'CSC'
    ax4 = fig.add_subplot(gs[1, 3])
    add_anova(ax4, v)

    # Add mean plots
    v = 'Wi'
    ax = fig.add_subplot(gs[0, 1])
    add_anova_mean(ax, v, ylim=ax1.get_ylim(), yticks=ax1.get_yticks(), show_xticks=False)
    v = 'ST'
    ax = fig.add_subplot(gs[1, 1])
    add_anova_mean(ax, v, ylim=ax2.get_ylim(), yticks=ax2.get_yticks())
    v = 'CF'
    ax = fig.add_subplot(gs[0, 4])
    add_anova_mean(ax, v, ylim=ax3.get_ylim(), yticks=ax3.get_yticks(), show_xticks=False)
    v = 'CSC'
    ax = fig.add_subplot(gs[1, 4])
    add_anova_mean(ax, v, ylim=ax4.get_ylim(), yticks=ax4.get_yticks())

    # Add section titles (in place of axes[0/1].set_title)
    fig.text(0.11, 0.99, "Policy-focused indicator", fontsize=12, va='top', ha='left')
    fig.text(0.534, 0.99, "Human behavioral variable", fontsize=12, va='top', ha='left')
    fig.text(-0.02, 0.5, "Fraction of variance explained", va='center', ha='center',
             rotation='vertical', fontsize=12)

    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    y1 = 0.93; y2 = 0.497
    positions = [
        (0.09, y1),  # for top-left subplot
        (0.404, y1),  # for top-right subplot
        (0.09, y2),  # for bottom-left subplot
        (0.404, y2),  # for bottom-right subplot
        (0.55, y1),  # for top-left subplot
        (0.862, y1),  # for top-right subplot
        (0.55, y2),  # for bottom-left subplot
        (0.862, y2),  # for bottom-right subplot
    ]

    for label, (x, y) in zip(labels, positions):
        fig.text(x, y, label, fontsize=10, fontweight='bold', ha='left', va='top')

    # Show the plot
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    if save_figname is not None:
        fig.savefig(save_figname, dpi=300, bbox_inches='tight')
    plt.show()

def plot_norm_comparison(pn, df_Wi_regime_norm, fraction=True, save_figname=None):
    def get_df_higher_lower(v_regime):
        if fraction:
            mu_dict_h = clt.io.read_pd_hdf5(file_path=pn.outputs.ANOVA.get()/f"anova_mu_fraction_{v_regime}_higher.h5")
            mu_dict_l = clt.io.read_pd_hdf5(file_path=pn.outputs.ANOVA.get()/f"anova_mu_fraction_{v_regime}_lower.h5")
        else:
            mu_dict_h = clt.io.read_pd_hdf5(file_path=pn.outputs.ANOVA.get()/f"anova_mu_sum_sq_{v_regime}_higher.h5")
            mu_dict_l = clt.io.read_pd_hdf5(file_path=pn.outputs.ANOVA.get()/f"anova_mu_sum_sq_{v_regime}_lower.h5")
        df_higher = []
        df_lower = []
        vlist = ['ST', 'CF', 'Wi', 'CSC']
        for v in vlist:
            df_higher.append(mu_dict_h[v].mean().to_frame(v))
            df_lower.append(mu_dict_l[v].mean().to_frame(v))
        df_higher = pd.concat(df_higher, axis=1).T
        df_lower = pd.concat(df_lower, axis=1).T
        return df_lower, df_higher

    def add_bars(ax, df_lower, df_higher, hatch_at="lower"):
        width = 0.35  # bar width
        gap = 0.08
        categories = df_higher.columns
        groups = df_higher.index  # e.g., ST, CF, Wi, CSC
        x = np.arange(len(groups))  # x positions for groups

        bottom = np.zeros(len(df_higher))
        for i, col in enumerate(categories):
            if hatch_at == "higher":
                ax.bar(x - width/2 - gap/2, df_higher[col], bottom=bottom, width=width,
                       hatch='//', label=col, color=reordered_colors[i], edgecolor='dimgray')
            else:
                ax.bar(x - width/2 - gap/2, df_higher[col], bottom=bottom, width=width,
                        label=col, color=reordered_colors[i], edgecolor='dimgray')
            bottom += df_higher[col]

        bottom = np.zeros(len(df_lower))
        for i, col in enumerate(categories):
            if hatch_at == "lower":
                ax.bar(x + width/2 + gap/2, df_lower[col], bottom=bottom, width=width,
                        hatch='//', color=reordered_colors[i], edgecolor='dimgray')
            else:
                ax.bar(x + width/2 + gap/2, df_lower[col], bottom=bottom, width=width,
                       color=reordered_colors[i], edgecolor='dimgray')
            bottom += df_lower[col]

        ax.set_xticks(x)
        ax.set_xticklabels([])  # Hide x-tick labels for top
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        if fraction:
            ax.set_ylim([0,1])

    def add_hbar(ax, df_regime_norm, reverse=False, hatch_at="lower"):
        df_plot = df_regime_norm.T

        # Prepare values
        vlist = ['ST', 'Wi', 'CF', 'CSC']
        y = np.arange(len(vlist))
        higher_vals = -df_plot.loc[vlist, 'higher'].values
        lower_vals = df_plot.loc[vlist, 'lower'].values  # make lower negative

        # Plot
        bar_width = 0.7
        cm = clt.cmap.Colormap('tol:pale')
        cm = [cm(i) for i in range(6)]
        if hatch_at == "lower":
            ax.barh(y, higher_vals, height=bar_width, label='Higher', color=cm, edgecolor='dimgray')
            ax.barh(y, lower_vals, height=bar_width, label='Lower', hatch='//', color=cm, edgecolor='dimgray')
        elif hatch_at == "higher":
            ax.barh(y, higher_vals, height=bar_width, label='Higher', hatch='//', color=cm, edgecolor='dimgray')
            ax.barh(y, lower_vals, height=bar_width, label='Lower', color=cm, edgecolor='dimgray')
        # Reverse the x-axis tick labels
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_xticklabels([1, 0.5, 0, 0.5, 1])

        # Add labels at center line
        for i, v in enumerate(vlist):
            # Label for lower (left side)
            if reverse:
                ax.text(0.02, y[i], var_dict[v], va='center', ha='left', fontsize=7)
            else:
                ax.text(-0.02, y[i], var_dict[v], va='center', ha='right', fontsize=7)

        # Aesthetics
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.axvline(0, color='black', linewidth=1)
        ax.set_xlim([-1, 1])
        ax.set_ylabel("Variable")

    # Create figure and GridSpec
    fig = plt.figure(figsize=(7, 2.5))
    gs = GridSpec(1, 2, width_ratios=[1.5, 1], height_ratios=[1], figure=fig)

    cm = clt.cmap.Colormap('petroff:petroff10')
    new_order = [0, 1, 5, 3, 4]  # Example: Custom order
    reordered_colors = [cm(i) for i in new_order]

    ##### First left subplot (top-left)
    ax = fig.add_subplot(gs[0, 0])
    v_regime = "Wi_regime"
    df_lower, df_higher = get_df_higher_lower(v_regime)
    add_bars(ax, df_lower, df_higher)
    ax.set_ylabel("Fraction of\nvariance explained", fontsize=10)
    ax.set_xticklabels([var_dict[v] for v in df_lower.index])
    ax.set_ylabel("Fraction of\nvariance explained", fontsize=10)

    ##### Second left subplot (bottom-left), share x-axis
    # ax = fig.add_subplot(gs[1, 0])
    # v_regime = "ST_regime"
    # df_lower, df_higher = get_df_higher_lower(v_regime)
    # add_bars(ax, df_lower, df_higher, hatch_at="higher")
    # ax.set_xticklabels([var_dict[v] for v in df_lower.index])
    # ax.set_ylabel("Fraction of\nvariance explained", fontsize=10)

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    map_dict = {"Pr": "Prec ratio\n(Pr)", "Cr": "Crop price\nratio (Cr)", "Co": "Corn field\nratio (Co)", "Residual": "Internal\nvariability", "Interaction terms": "Interaction\nterms"}
    labels = [map_dict[i] if map_dict.get(i) is not None else i for i in labels]
    legend1 = fig.legend(handles, labels, ncols=5, bbox_to_anchor=(0.5, 0.03), loc='upper center', frameon=False, fontsize=8)
    # Second legend: hatch styles
    hatch_lower = mpatches.Patch(facecolor='white', edgecolor='dimgray', hatch='//', label='Lower than\nhistorical values')
    hatch_higher = mpatches.Patch(facecolor='white', edgecolor='dimgray', hatch=None, label='Higher than\nhistorical values')
    legend2 = fig.legend(
        [hatch_higher, hatch_lower],
        ['Higher irrigation norm', 'Lower irrigation norm'],
        ncol=2,
        bbox_to_anchor=(0.5, -0.1),
        loc='upper center',
        frameon=False,
        fontsize=8
    )

    # Ensure both legends are added to the figure
    fig.add_artist(legend1)
    fig.add_artist(legend2)

    ##### Third right subplot (top-right)
    ax = fig.add_subplot(gs[0, 1])
    add_hbar(ax, df_Wi_regime_norm, reverse=False)
    ax.set_xlabel('Normalized mean value')

    ##### Forth right subplot (bottom-right)
    # ax = fig.add_subplot(gs[1, 1])
    # add_hbar(ax, df_ST_regime_norm, reverse=True, hatch_at="higher")
    # ax.set_xlabel('Normalized mean value')

    labels = ['(a)', '(b)']
    positions = [
        (0.01, 0.99),  # for top-left subplot
        (0.61, 0.99),  # for top-right subplot
    ]

    for label, (x, y) in zip(labels, positions):
        fig.text(x, y, label, fontsize=10, fontweight='bold', ha='left', va='top')

    #fig.text(-0.03, 0.78, "Withdrawal regime\ncomparison", fontsize=10, ha='center', va='center', rotation=90)
    #fig.text(-0.03, 0.32, "Saturated thickness\nregime comparison", fontsize=10, ha='center', va='center', rotation=90)

    # Adjust layout
    fig.tight_layout()
    if save_figname is not None:
        fig.savefig(save_figname, dpi=300, bbox_inches='tight')
    plt.show()


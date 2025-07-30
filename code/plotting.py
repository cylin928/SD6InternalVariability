import os
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns
import re


class SD6Visual:
    def __init__(
        self,
        output_dir: str = None,
        extension: str = ".jpg",
        show_fig: bool = True,
    ):
        # Output settings
        self.output_dir = output_dir
        self.extension = extension
        self.show_fig = show_fig

        # Color settings
        self.colors = {
            "dry_year": "gainsboro",
            "GW_st": "#4472C4",
            "withdrawal": "#4472C4",
            "Well": "#C00000",
            "Profit": "#FFC000",  # for profit and crop price and eco
            "Imitation": "#17BECF",
            "Social comparison": "#E377C2",
            "Repetition": "#FF7F0E",
            "Deliberation": "#9467BD",
            "others": "grey",
            "corn": "#F6BB00",  # for corn ratio & glm model
            "prec": "dodgerblue",  # for precipitation
            "SD6Model": "slategray",  # for SD6 model
            "crop_price": "saddlebrown",  # for crop price
            "set-prec_id": "mediumaquamarine",
            "set-eco_id": "peru",
            "set-py_seed": "tan",
            "prec_id-eco_id": "darkcyan",
            "prec_id-py_seed": "lightsteelblue",
            "eco_id-py_seed": "darkgoldenrod",
        }

        # Figure width settings
        self.figwd = {
            "1": 140 / 1.5 / 25.4,
            "1.5": 140 / 25.4,
            "2": 140 / 1.5 / 25.4 * 2,
        }

        # Plot settings
        self.fontsize = 16
        plt.rcParams["font.size"] = self.fontsize  # sets the default font size
        plt.rcParams["font.family"] = "Arial"  # sets the default font family

        # add default sd6 plotting info
        self.add_sd6_plotting_info()

    def add_sd6_plotting_info(
        self,
        sd6_data: pd.DataFrame = None,
        prec_avg: pd.DataFrame = None,
        year_range: list = [2012, 2022],
        cali_years: list = [2013, 2019],
        vali_years: list = [2020, 2022],
        lema_years: list = [2013, 2018],
        dry_years: list | None = None,
    ) -> None:
        """
        Add the common plotting information for the SD6 model.

        Parameters
        ----------
        sd6_data : pd.DataFrame, optional
            The SD6 observation, by default None
        prec_avg : pd.DataFrame, optional
            The average precipitation data, by default None
        year_range : list, optional
            The year range to plot, by default [2012, 2022]
        cali_years : list, optional
            The calibration period, by default [2013, 2019]
        vali_years : list, optional
            The validation period, by default [2020, 2022]
        lema_years : list, optional
            The LEMA period, by default [2013, 2018]
        dry_years : list, optional
            The dry years, by default None (will be calculated from the precipitation
            data using the mean precipitation as the threshold)

        Returns
        -------
        None
        """

        @dataclass
        class SD6PlottingInfo:
            pass

        sd6_plotting_info = SD6PlottingInfo()

        # Model outputs
        sd6_plotting_info.sd6_data = (
            sd6_data.loc[year_range[0] : year_range[1]]
            if sd6_data is not None
            else None
        )
        sd6_plotting_info.prec_avg = prec_avg
        sd6_plotting_info.year_range = year_range
        sd6_plotting_info.cali_years = cali_years
        sd6_plotting_info.vali_years = vali_years
        sd6_plotting_info.lema_years = lema_years
        sd6_plotting_info.dry_years = dry_years

        if sd6_plotting_info.prec_avg is not None:
            sd6_plotting_info.prec_avg["year"] = sd6_plotting_info.prec_avg.index
            sd6_plotting_info.wet_dry_threshold = prec_avg["annual"].mean()
        self.sd6_plotting_info = sd6_plotting_info

    def save_figure(self, fig: plt.Figure, name: str) -> None:
        """
        Save the figure to the output directory.

        Parameters
        ----------
        fig : plt.Figure
            The figure to save.
        name : str
            The name of the figure.

        Returns
        -------
        None
        """
        if self.output_dir:
            if "." in name:
                filename = os.path.join(self.output_dir, name)
            else:
                filename = os.path.join(self.output_dir, name, self.extension)
            fig.savefig(
                filename,
                dpi=500,
                bbox_inches="tight",
            )
        else:
            raise ValueError("Output directory is not specified.")

        if self.show_fig:
            plt.show()

    def add_dry_year_regions(self, ax: Axes) -> None:
        """Add the dry year regions to the plot."""
        info = self.sd6_plotting_info
        for _, row in info.prec_avg.iterrows():
            if row["annual"] <= info.wet_dry_threshold:
                ax.axvspan(
                    row["year"] - 0.5,
                    row["year"] + 0.5,
                    color=self.colors["dry_year"],
                    lw=0,
                    zorder=0,
                )

    def add_LEMA_vlines(self, ax: Axes) -> None:
        """Add the vertical lines to represent the start of LEMA in the plot."""
        info = self.sd6_plotting_info
        for year in info.lema_years:
            ax.axvline(year, c="grey", ls=":", lw=1, zorder=1)

    def add_cali_vali_vlines(self, ax: Axes) -> None:
        """Add the vertical lines to represent the start of calibration and
        validation in the plot."""
        info = self.sd6_plotting_info
        ax.axvline(info.cali_years[0], c="k", ls="-.", lw=1.1, zorder=1)
        ax.axvline(info.vali_years[0], c="k", ls="-.", lw=1.1, zorder=1)

    def auto_ylim(self, ylabel: str | None, ax: Axes) -> None:
        """Automatically set the y-axis limits based on the ylabel."""
        if ylabel:
            if "ratio" in ylabel:
                ax.set_ylim([-0.01, 1])

    def add_prec_bars_on_secondary_yaxis(
        self, ax: Axes, scale: float = 0.1, invert_yaxis: bool = True
    ) -> None:
        """Add the precipitation bars on the secondary y-axis."""
        info = self.sd6_plotting_info
        ax2 = ax.twinx()
        ax2.bar(
            info.prec_avg["year"],
            info.prec_avg["annual"],
            color=self.colors["prec"],
            alpha=0.5,
            width=0.95,
            zorder=0,
        )
        # ax2.set_ylabel("Precipitation (mm)")
        ax2.set_ylim([0, max(info.prec_avg["annual"]) / scale])
        if invert_yaxis:
            ax2.invert_yaxis()
        ax2.set_yticks([])  # This removes the ticks
        ax2.set_yticklabels([])  # This ensures no labels are shown

    def add_filled_rectangle_with_text(
        self,
        ax: Axes,
        text: str,
        x0: float,
        y0: float,
        width: float,
        height: float,
        edgecolor: str = "k",
        color: str = "none",
        fontsize: float = 11,
        zorder: int = 100,
    ):
        # Create a rectangle
        rect = Rectangle(
            (x0, y0),
            width,
            height,
            linewidth=0.7,
            edgecolor=edgecolor,
            facecolor=color,
            zorder=zorder,
        )
        ax.add_patch(rect)
        # Add text in the center of the rectangle
        ax.text(
            x0 + width / 2,
            y0 + height / 2 * 0.9,
            text,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=fontsize,
            zorder=zorder,
        )

    def plot_timeseries(
        self,
        df_sys: pd.DataFrame,
        stochastic_df_list: list = [],
        metrices: pd.DataFrame | None = None,
        show_date: bool = True,
        show_stochastic: bool = True,
        add_LEMA_vlines: bool = True,
        add_cali_vali: bool = False,
        add_dry_year_regions: bool = True,
        add_prec_bars_on_secondary_yaxis: bool = True,
        fig_name: str | None = None,
    ):
        info = self.sd6_plotting_info
        df_sys = df_sys.loc[info.year_range[0] : info.year_range[1]]

        fontsize = 16
        plt.rcParams["font.size"] = fontsize

        fig = plt.figure(figsize=(self.figwd["1.5"], self.figwd["1.5"] * 1.3))
        axes = []
        x = list(df_sys.index)

        lw_sim = 2.5
        lw_obv = 1
        lw_sto = 0.7
        ylabel_xloc = -0.12

        def add_legend(ax):
            ax.plot(
                [0] * len(df_sys[var_]),
                df_sys[var_],
                label="Sim",
                lw=lw_sim,
                c="k",
            )  # pseudo line
            handles, labels = ax.get_legend_handles_labels()
            rect = Patch(
                facecolor=self.colors["dry_year"], edgecolor=None, label="Dry year"
            )
            handles.append(rect)
            labels.append("Dry year")
            ax.legend(
                handles,
                labels,
                loc="upper right",
                ncols=3,
                bbox_to_anchor=(1, 1.3),
                frameon=False,
            )

        def plot_var_timeseries(
            ax, var_: str, var_rescale: float = 1.0, ylabel: str = ""
        ):
            ax.plot(
                x,
                df_sys[var_] * var_rescale,
                c=self.colors[var_],
                lw=lw_sim,
                zorder=10,
            )
            if show_stochastic:
                for df in stochastic_df_list:
                    df = df.loc[info.year_range[0] : info.year_range[1]]
                    ax.plot(
                        x,
                        df[var_] * var_rescale,
                        c=self.colors[var_],
                        alpha=0.3,
                        lw=lw_sto,
                        zorder=5,
                    )
            if show_date and info.sd6_data is not None:
                ax.plot(
                    x,
                    info.sd6_data[var_] * var_rescale,
                    c="k",
                    ls="--",
                    label="Obv",
                    lw=lw_obv,
                    zorder=20,
                )
            if metrices:
                ax.text(
                    0.05,
                    0.05,
                    f"RMSE: {round(metrices.loc[var_, 'rmse'], 3)}",
                    transform=ax.transAxes,
                    verticalalignment="bottom",
                    horizontalalignment="left",
                )
            if add_dry_year_regions:
                self.add_dry_year_regions(ax)
            if add_LEMA_vlines:
                self.add_LEMA_vlines(ax)
            ax.set_xlim(info.year_range)
            ax.set_ylabel(ylabel)
            ax.yaxis.set_label_coords(ylabel_xloc, 0.5)

        # Saturated thickness
        var_ = "GW_st"
        ax = fig.add_axes([0, 1 / 4 * 4 + 1 / 4 / 3 * 2.5, 1, 1 / 4])
        axes.append(ax)
        plot_var_timeseries(ax, var_, ylabel="(a)\nSaturated\nthickness\n(m)")
        add_legend(ax)  # Just for the first subplot
        ax.tick_params(
            axis="x", which="both", bottom=True, top=False, labelbottom=False
        )
        ax.set_ylim([16.9, 19.5])
        ax.tick_params(
            axis="x",
            which="both",
            bottom=True,
            top=False,
            labelbottom=False,
            direction="in",
        )
        if add_cali_vali:
            self.add_filled_rectangle_with_text(
                ax, "warm-up", 2012, 19.2, 1, 0.3, fontsize=9.5
            )
            self.add_filled_rectangle_with_text(ax, "calibration", 2013, 19.2, 6, 0.3)
            self.add_filled_rectangle_with_text(ax, "validation", 2019, 19.2, 3, 0.3)

        # Withdrawal
        var_ = "withdrawal"
        ax = fig.add_axes([0, 1 / 4 * 3 + 1 / 4 / 3 * 2.5, 1, 1 / 4])
        axes.append(ax)
        plot_var_timeseries(
            ax, var_, var_rescale=1, ylabel="(b)\n\nWithdrawal\n($10^6$ $m^3$)"
        )
        ax.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        ax.set_ylim([9, 45])
        ax.tick_params(
            axis="x",
            which="both",
            bottom=True,
            top=False,
            labelbottom=False,
            direction="in",
        )
        if add_prec_bars_on_secondary_yaxis:
            self.add_prec_bars_on_secondary_yaxis(ax, invert_yaxis=False)

        # CONSUMAT state ratios
        ax = fig.add_axes([0, 4 / 7, 1, 1 / 4 + 1 / 4 / 8])
        axes.append(ax)
        states = ["Imitation", "Social comparison", "Repetition", "Deliberation"]
        dff = df_sys[states]
        num_agt = sum(dff.iloc[0, :])
        dff = dff / num_agt
        for state in states:
            ax.plot(
                x, dff[state], zorder=20, label=state, c=self.colors[state], lw=lw_sim
            )
        if show_stochastic:
            for df in stochastic_df_list:
                df = df.loc[info.year_range[0] : info.year_range[1]]
                dff = df[states]
                dff = dff / num_agt
                for state in states:
                    ax.plot(
                        x,
                        dff[state],
                        c=self.colors[state],
                        alpha=0.3,
                        lw=lw_sto,
                        zorder=5,
                    )
        if add_dry_year_regions:
            self.add_dry_year_regions(ax)
        if add_LEMA_vlines:
            self.add_LEMA_vlines(ax)
        ax.set_xlim(info.year_range)
        ax.set_ylim([-0.02, 1])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_ylabel("(c)\nCONSUMAT\nstate ratio\n(-)")
        ax.set_xlabel("Year")
        ax.legend(
            ncols=2,
            labelspacing=0.1,
            loc="upper right",
            bbox_to_anchor=(1, 1.35),
            frameon=False,
        )
        ax.yaxis.set_label_coords(ylabel_xloc, 0.5)
        plt.tight_layout()

        if fig_name is not None:
            if fig_name == "":
                fig_name = f"sd6_ts_GW_st-withdrawal-consumat.jpg"
            self.save_figure(fig, fig_name)
        else:
            plt.show()

    def plot_st_withdrawal_corn_timeseries(
        self,
        df_sys: pd.DataFrame,
        stochastic_df_list: list = [],
        metrices: pd.DataFrame | None = None,
        show_date: bool = True,
        show_stochastic: bool = True,
        add_LEMA_vlines: bool = True,
        add_cali_vali: bool = False,
        add_dry_year_regions: bool = True,
        add_prec_bars_on_secondary_yaxis: bool = True,
        fig_name: str | None = None,
    ):
        info = self.sd6_plotting_info
        df_sys = df_sys.loc[info.year_range[0] : info.year_range[1]]

        fontsize = 16
        plt.rcParams["font.size"] = fontsize

        fig = plt.figure(figsize=(self.figwd["1.5"], self.figwd["1.5"] * 1))
        axes = []
        x = list(df_sys.index)

        lw_sim = 2.5
        lw_obv = 1
        lw_sto = 0.7
        ylabel_xloc = -0.12

        def add_legend(ax):
            ax.plot(
                [0] * len(df_sys[var_]),
                df_sys[var_],
                label="Observed",
                lw=lw_sim,
                c="k",
                ls="--"
            )  # pseudo line
            ax.plot(
                [0] * len(df_sys[var_]),
                df_sys[var_],
                label="Simulated",
                lw=lw_sim,
                c="k",
            )  # pseudo line
            handles, labels = ax.get_legend_handles_labels()
            rect = Patch(
                facecolor=self.colors["dry_year"], edgecolor=None, label="Dry year"
            )
            handles.append(rect)
            labels.append("Dry year")
            ax.legend(
                handles,
                labels,
                loc="upper right",
                ncols=3,
                bbox_to_anchor=(1.03, 1.33),
                frameon=False,
            )

        def plot_var_timeseries(
            ax, var_: str, var_rescale: float = 1.0, ylabel: str = ""
        ):
            ax.plot(
                x,
                df_sys[var_] * var_rescale,
                c=self.colors[var_],
                lw=lw_sim,
                zorder=10,
            )
            if show_stochastic:
                for df in stochastic_df_list:
                    df = df.loc[info.year_range[0] : info.year_range[1]]
                    ax.plot(
                        x,
                        df[var_] * var_rescale,
                        c=self.colors[var_],
                        alpha=0.3,
                        lw=lw_sto,
                        zorder=5,
                    )
            if show_date and info.sd6_data is not None:
                ax.plot(
                    x,
                    info.sd6_data[var_] * var_rescale,
                    c="k",
                    ls="--",
                    label="Observed",
                    lw=lw_obv,
                    zorder=20,
                )
            if metrices:
                ax.text(
                    0.05,
                    0.05,
                    f"RMSE: {round(metrices.loc[var_, 'rmse'], 3)}",
                    transform=ax.transAxes,
                    verticalalignment="bottom",
                    horizontalalignment="left",
                )
            if add_dry_year_regions:
                self.add_dry_year_regions(ax)
            if add_LEMA_vlines:
                self.add_LEMA_vlines(ax)
            ax.set_xlim([2013, 2022])
            ax.set_ylabel(ylabel)
            ax.yaxis.set_label_coords(ylabel_xloc, 0.5)

        # Withdrawal
        var_ = "withdrawal"
        ax = fig.add_axes([0, 1 / 4 * 4 + 1 / 4 / 3 * 2.5, 1, 1 / 4])
        axes.append(ax)
        add_legend(ax)  # Just for the first subplot
        plot_var_timeseries(
            ax, var_, var_rescale=0.01, ylabel="\nWithdrawal\n($10^6$ $m^3$)"
        )
        ax.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        ax.set_ylim([9, 40])
        ax.tick_params(
            axis="x",
            which="both",
            bottom=True,
            top=False,
            labelbottom=False,
            direction="in",
        )
        if add_prec_bars_on_secondary_yaxis:
            self.add_prec_bars_on_secondary_yaxis(ax, invert_yaxis=False)

        # Saturated thickness
        var_ = "GW_st"
        ax = fig.add_axes([0, 1 / 4 * 3 + 1 / 4 / 3 * 2.5, 1, 1 / 4])
        axes.append(ax)
        plot_var_timeseries(ax, var_, ylabel="Saturated\nthickness\n(m)")
        ax.tick_params(
            axis="x", which="both", bottom=True, top=False, labelbottom=False
        )
        ax.set_ylim([16.5, 18.7])
        ax.tick_params(
            axis="x",
            which="both",
            bottom=True,
            top=False,
            labelbottom=False,
            direction="in",
        )
        if add_cali_vali:
            self.add_filled_rectangle_with_text(
                ax, "warm-up", 2012, 19.2, 1, 0.3, fontsize=9.5
            )
            self.add_filled_rectangle_with_text(ax, "calibration", 2013, 19.2, 6, 0.3)
            self.add_filled_rectangle_with_text(ax, "validation", 2019, 19.2, 3, 0.3)

        # Corn field ratio
        var_ = "corn"
        ax = fig.add_axes([0, 1 / 4 * 2 + 1 / 4 / 3 * 2.5, 1, 1 / 4])
        axes.append(ax)
        plot_var_timeseries(
            ax, var_, var_rescale=1, ylabel="Field ratio\nfor corn\n(--)"
        )

        ax.set_ylim([0, 1])
        ax.set_xlabel("Year")

        ax.yaxis.set_label_coords(ylabel_xloc, 0.5)
        plt.tight_layout()

        if fig_name is not None:
            if fig_name == "":
                fig_name = f"sd6_ts_GW_st-withdrawal-corn.jpg"
            self.save_figure(fig, fig_name)
        else:
            plt.show()



    def plot_crop_ratio(
        self,
        df_sys: pd.DataFrame,
        stochastic_df_list: list = [],
        metrices: pd.DataFrame | None = None,
        show_data: bool = True,
        show_stochastic: bool = True,
        crop_options=["corn", "others"],
        add_LEMA_vlines: bool = True,
        add_cali_vali: bool = False,
        add_dry_year_regions: bool = True,
        add_prec_bars_on_secondary_yaxis: bool = True,
        fig_name: str | None = None,
    ):
        info = self.sd6_plotting_info
        df_sys = df_sys.loc[info.year_range[0] : info.year_range[1]]

        fontsize = 10.8
        plt.rcParams["font.size"] = fontsize

        abcde = ["a", "b", "c", "d", "e"]

        fig, axes = plt.subplots(
            ncols=1,
            nrows=2,
            figsize=(self.figwd["1.5"], self.figwd["1"] * 1.2),
            sharex=True,
            sharey=True,
        )
        axes = axes.flatten()
        for i, crop in enumerate(crop_options):
            ax = axes[i]
            x = list(df_sys.index)
            if metrices:
                rmse = round(metrices.loc[crop, "rmse"], 3)
                if crop == "corn":
                    ax.text(
                        0.05,
                        0.05,
                        f"RMSE: {rmse}",
                        transform=ax.transAxes,
                        verticalalignment="bottom",
                        horizontalalignment="left",
                    )
                else:
                    ax.text(
                        0.05,
                        0.80,
                        f"RMSE: {rmse}",
                        transform=ax.transAxes,
                        verticalalignment="bottom",
                        horizontalalignment="left",
                    )

            ax.plot(x, df_sys[crop], c=self.colors[crop], zorder=100, lw=2)
            if show_data and info.sd6_data is not None:
                ax.plot(x, info.sd6_data[crop], c="k", ls="--", zorder=1000, lw=1)

            if show_stochastic:
                for df in stochastic_df_list:
                    df = df.loc[info.year_range[0] : info.year_range[1]]
                    ax.plot(
                        x, df[crop], c=self.colors[crop], zorder=1, alpha=0.2, lw=0.5
                    )

            ax.set_ylabel(
                f"({abcde[i]})\n" + crop.capitalize() + " ratio", fontsize=fontsize
            )
            ax.set_ylim([-0.01, 1])
            if add_LEMA_vlines:
                self.add_LEMA_vlines(ax)
            if add_dry_year_regions:
                self.add_dry_year_regions(ax)
            ax.set_xlim(info.year_range)
            if add_cali_vali:
                self.add_filled_rectangle_with_text(
                    ax, "warm-up", 2012, 0.9, 1, 0.1, fontsize=6.2
                )
                self.add_filled_rectangle_with_text(
                    ax, "calibration", 2013, 0.9, 6, 0.1, fontsize=8
                )
                self.add_filled_rectangle_with_text(
                    ax, "validation", 2019, 0.9, 3, 0.1, fontsize=8
                )
            if add_prec_bars_on_secondary_yaxis:
                self.add_prec_bars_on_secondary_yaxis(ax, invert_yaxis=False)

        fig.add_subplot(111, frameon=False)
        plt.tick_params(
            labelcolor="none",
            which="both",
            top=False,
            bottom=False,
            left=False,
            right=False,
        )
        plt.xlabel("Year")
        line_obv = Line2D([0], [0], label="Obv", c="k", ls="--")
        line_sim = Line2D([0], [0], label="Sim", c="k", ls="-")
        rect = Patch(
            facecolor=self.colors["dry_year"], edgecolor=None, label="Dry year"
        )

        plt.legend(
            handles=[line_obv, line_sim, rect],
            labelspacing=0.8,
            frameon=False,
            ncol=3,
            loc="upper right",
            bbox_to_anchor=(1, 1.1),
        )
        plt.tight_layout()

        if fig_name is not None:
            if fig_name == "":
                fig_name = f"sd6_ts_crop_ratio.jpg"
            self.save_figure(fig, fig_name)
        else:
            plt.show()

    def plot_sns_lineplot_for_task2(
        self,
        df_ts: pd.DataFrame,
        var_name: str,
        fig_name: str | None = None,
        add_LEMA_vlines: bool = True,
        add_dry_year_regions: bool = True,
        add_prec_bars_on_secondary_yaxis: bool = True,
        ylabel: str | None = None,
        **kwargs,
    ) -> None:
        fontsize = 10
        plt.rcParams["font.size"] = fontsize
        fig, ax = plt.subplots(figsize=(self.figwd["2"], self.figwd["2"] * 0.55))
        hue_order = sorted(df_ts["ratio_group"].unique())
        style_order = sorted(df_ts["range"].unique())
        sns.lineplot(
            data=df_ts,
            x="year",
            y=var_name,
            hue="ratio_group",
            style="range",
            hue_order=hue_order,
            style_order=style_order,
            ax=ax,
            **kwargs,
        )
        if add_LEMA_vlines:
            self.add_LEMA_vlines(ax)
        if add_dry_year_regions:
            self.add_dry_year_regions(ax)
        if add_prec_bars_on_secondary_yaxis:
            if var_name in ["Repetition", "corn", "GW_st"]:
                self.add_prec_bars_on_secondary_yaxis(ax, invert_yaxis=False)
            else:
                self.add_prec_bars_on_secondary_yaxis(ax, invert_yaxis=True)
        info = self.sd6_plotting_info
        ax.set_xlim(info.year_range)
        if ylabel:
            ax.set_ylabel(ylabel)
        self.auto_ylim(ylabel, ax)
        ax.set_xlabel("Year")
        # Create custom legend
        handles, labels = ax.get_legend_handles_labels()
        # Assuming the first two handles and labels are for 'ratio_group' and the rest are for 'range'
        legend_ratio = ax.legend(
            handles=handles[1 : len(hue_order) + 1],
            labels=labels[1 : len(hue_order) + 1],
            title="Corn ratio\ngroup (%)",
            bbox_to_anchor=(1.07, 0.47),
            loc="center left",
            borderaxespad=0.0,
            ncol=1,
            frameon=False,
        )
        ax.add_artist(legend_ratio)
        ax.legend(
            handles=handles[len(hue_order) + 2 :],
            labels=labels[len(hue_order) + 2 :],
            title="Range (km)",
            bbox_to_anchor=(1.31, 0.5),
            loc="center left",
            borderaxespad=0.0,
            ncol=1,
            frameon=False,
        )

        # Show the plot
        plt.tight_layout()  # Adjust the layout to make room for the legend

        if fig_name is not None:
            if fig_name == "":
                fig_name = f"task2_ts_{var_name}.jpg"
            self.save_figure(fig, fig_name)
        else:
            plt.show()

    def plot_sns_heatmap_for_task2(
        self,
        df_metrices: pd.DataFrame,
        var_name: str,
        period: str,
        do_row_normalization: bool,
        palette: str = "viridis",
        fig_name: str = None,
        legend_title: str | None = "",
        xlabel: str | None = "Range (km)",
        ylabel: str | None = "Corn ratio group (%)",
        **kwargs,
    ) -> None:
        mask = (df_metrices["variable"] == var_name) & (df_metrices["period"] == period)
        table_axises = ["ratio_group", "range", "value"]
        df = df_metrices.loc[mask, table_axises]
        dff = df.groupby(table_axises[:2]).mean().reset_index()
        table = dff.pivot(index="ratio_group", columns="range", values="value")

        fontsize = 9
        plt.rcParams["font.size"] = fontsize
        fig, ax = plt.subplots(figsize=(self.figwd["1"], self.figwd["1"] * 0.9))

        # Create a colormap that sets NaN values to grey
        cmap = sns.color_palette(palette, as_cmap=True)
        cmap.set_bad("grey")  # 'bad' data points (NaNs) are grey
        if legend_title != "":
            cbar_kws = {"extend": "neither", "label": legend_title}
        elif legend_title == "":
            cbar_kws = {"extend": "neither", "label": f"{var_name} [{period}]"}

        # Create the heatmap
        if not do_row_normalization:
            sns.heatmap(table, cmap=cmap, cbar_kws=cbar_kws, ax=ax, **kwargs)
            row_type = "standard"
        else:
            table_row_normalized = table.sub(table.mean(axis=1), axis=0)
            sns.heatmap(
                table_row_normalized,
                cmap=cmap,
                cbar_kws=cbar_kws,
                ax=ax,
                **kwargs,
            )
            row_type = "row_normalized"

        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        if fig_name is not None:
            if fig_name == "":
                fig_name = f"task2_heatmap_{var_name}-{period}_{row_type}.jpg"
            self.save_figure(fig, fig_name)
        else:
            plt.show()

    def plot_sns_boxplot_for_task2(
        self,
        df_metrices: pd.DataFrame,
        var_name: str,
        period_list: str,
        point_statistic: str,
        label_max_or_min: str,
        fig_name: str = None,
        ylabel: str | None = None,
        **kwargs,
    ) -> None:
        mask = (df_metrices["variable"] == var_name) & df_metrices["period"].isin(
            period_list
        )
        df = df_metrices[mask]
        # Create the boxplot
        fontsize = 10
        plt.rcParams["font.size"] = fontsize
        fig, ax = plt.subplots(figsize=(self.figwd["1.5"], self.figwd["2"] * 0.7))
        sns.boxplot(
            data=df, x="range", y="value", hue="period", ax=ax, zorder=1, **kwargs
        )
        ax.set_ylabel(var_name)

        # Calculate means
        statistic_values = df.groupby(["range", "period"])["value"]
        statistic_values = eval(f"statistic_values.{point_statistic}().reset_index()")

        # Colors corresponding to the periods
        palette = sns.color_palette("tab10", n_colors=len(df["period"].unique()))

        # Number of unique periods
        periods = df["period"].unique()
        n_periods = len(periods)
        width = 0.8  # Total width of all boxes in one group
        range_order = sorted(df["range"].unique())

        # Add scatter plot for statistic values with adjusted positions
        for idx, period in enumerate(periods):
            period_means = statistic_values[statistic_values["period"] == period]
            # Calculate offsets
            offset = ((width / n_periods) * idx) - (width / n_periods) * (
                n_periods - 1
            ) / 2
            # Adjust x positions
            x_positions = [x + offset for x in range(len(period_means["range"]))]
            ax.scatter(
                x_positions,
                period_means["value"],
                color=palette[idx],
                s=30,
                edgecolor="k",
                label=f"{point_statistic.capitalize()} {period}",
                zorder=20,
            )

        # Connect statistic points with lines
        for idx, period in enumerate(periods):
            period_means = statistic_values[
                statistic_values["period"] == period
            ].sort_values(by="range")
            x_positions = [
                x
                + ((width / n_periods) * idx)
                - (width / n_periods) * (n_periods - 1) / 2
                for x in range(len(period_means["range"]))
            ]
            ax.plot(
                x_positions,
                period_means["value"],
                color=palette[idx],
                lw=0.5,
                zorder=10,
            )

        # Highlight the minimum mean value's marker frame for each period
        for idx, period in enumerate(periods):
            period_means = statistic_values[statistic_values["period"] == period]
            min_mean = period_means[
                period_means["value"]
                == eval(f"period_means['value'].{label_max_or_min}()")
            ]
            x = range_order.index(min_mean["range"].values[0])
            x_positions = [
                x
                + ((width / n_periods) * idx)
                - (width / n_periods) * (n_periods - 1) / 2
            ]
            ax.scatter(
                x_positions,
                min_mean["value"],
                color=palette[idx],
                s=30,
                marker="D",
                edgecolor="k",
                linewidth=2,
                label=f"{label_max_or_min.capitalize()} {point_statistic.capitalize()} {period}",
                zorder=30,
            )

        if ylabel:
            ax.set_ylabel(ylabel)
        ax.set_xlabel("Range (km)")

        # Show the plot
        plt.legend(
            bbox_to_anchor=(1, 0.5),
            loc="center left",
            borderaxespad=0.0,
            ncol=1,
            frameon=False,
        )

        if fig_name is not None:
            if fig_name == "":
                fig_name = f"task2_boxplot_{var_name}_{point_statistic}_{'-'.join(period_list)}.jpg"
            self.save_figure(fig, fig_name)
        else:
            plt.show()

    def plot_init_corn_spatial_distribution(
        self, init_corn: list, x: list, y: list
    ) -> None:
        fontsize = 12
        plt.rcParams["font.size"] = fontsize

        cmap = ListedColormap([self.colors["others"], self.colors["corn"]])
        fig, ax = plt.subplots(figsize=(self.figwd["1.5"], self.figwd["1"] * 1.1))
        ax.scatter(x, y, c=init_corn, cmap=cmap, s=50)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Create custom legend
        legend_handles = [
            Patch(color=self.colors["corn"], label="Corn"),
            Patch(color=self.colors["others"], label="Others"),
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            frameon=False,
            ncol=2,
            bbox_to_anchor=(0.0, 1.1),
        )

        plt.tight_layout()
        plt.show()

    def plot_sobol_first_and_second_order_indices(
        self, df_s12, xlabel: str = "Year", fig_name: str | None = None
    ):
        un_source_colors = {
            "set": self.colors["corn"],
            "prec_id": self.colors["prec"],
            "eco_id": self.colors["crop_price"],
            "py_seed": self.colors["SD6Model"],
            "set-prec_id": self.colors["set-prec_id"],
            "set-eco_id": self.colors["set-eco_id"],
            "set-py_seed": self.colors["set-py_seed"],
            "prec_id-eco_id": self.colors["prec_id-eco_id"],
            "prec_id-py_seed": self.colors["prec_id-py_seed"],
            "eco_id-py_seed": self.colors["eco_id-py_seed"],
            "SpatialGLM": self.colors["corn"],
            "Precipitation": self.colors["prec"],
            "CropPrice": self.colors["crop_price"],
            "SD6Model": self.colors["SD6Model"],
            "Precipitation-SpatialGLM": self.colors["set-prec_id"],
            "CropPrice-SpatialGLM": self.colors["set-eco_id"],
            "SpatialGLM-SD6Model": self.colors["set-py_seed"],
            "Precipitation-CropPrice": self.colors["prec_id-eco_id"],
            "Precipitation-SD6Model": self.colors["prec_id-py_seed"],
            "CropPrice-SD6Model": self.colors["eco_id-py_seed"],
        }

        colors = [un_source_colors[col] for col in df_s12.columns]

        fontsize = 12
        plt.rcParams["font.size"] = fontsize

        fig, ax = plt.subplots(figsize=(self.figwd["2"], self.figwd["1"] * 1.1))
        # if xlabel=="Year":
        #    df_s12.plot(kind="area", stacked=True, ax=ax, color=colors)
        # else:
        df_s12.plot(kind="bar", stacked=True, ax=ax, color=colors)
        ax.axhline(0, c="k", lw=0.5, ls="--")
        ax.axhline(1, c="k", lw=0.5, ls="--")

        if xlabel == "Year":
            pattern = re.compile(r"\b\d{4}\b")
            # Extract the year from each string in the list
            # xlabels = list(df_s12.index)
            xlabels = [label.get_text() for label in ax.get_xticklabels()]
            years = [
                pattern.search(date_string).group()
                for date_string in xlabels
                if pattern.search(date_string)
            ]
            pattern = re.compile(r"\(.*?\)")
            var_name = pattern.sub("", xlabels[0])
            ax.set_xticklabels(years, rotation=0)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(f"Variance contribution on {var_name}")
            ax.legend(
                title="Uncertainty sources",
                bbox_to_anchor=(1.01, 0.5),
                loc="center left",
                frameon=False,
            )
        else:
            var_name = "peroids"
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Variance contribution")
            # ax.tick_params(axis='x', labelrotation=60)
            ax.legend(
                title="Uncertainty sources",
                bbox_to_anchor=(1.01, 0.3),
                loc="center left",
                frameon=False,
            )

        plt.tight_layout()

        if fig_name is not None:
            if fig_name == "":
                fig_name = f"task3_sobel_12_{var_name}_{xlabel}.jpg"
            self.save_figure(fig, fig_name)
        else:
            plt.show()

    def plot_sobo_total_effect_indice(
        self, df_st, xlabel: str = "Year", fig_name: str | None = None
    ):
        un_source_colors = {
            "set": self.colors["corn"],
            "prec_id": self.colors["prec"],
            "eco_id": self.colors["crop_price"],
            "py_seed": self.colors["SD6Model"],
            "SpatialGLM": self.colors["corn"],
            "Precipitation": self.colors["prec"],
            "CropPrice": self.colors["crop_price"],
            "SD6Model": self.colors["SD6Model"],
        }

        colors = [un_source_colors[col] for col in df_st.columns]

        df_st = df_st.copy()
        df_st = df_st.divide(df_st.sum(axis=1), axis=0) * 100
        fontsize = 12
        plt.rcParams["font.size"] = fontsize

        fig, ax = plt.subplots(figsize=(self.figwd["2"], self.figwd["1"] * 1.1))
        if xlabel == "Year":
            df_st.plot(kind="area", stacked=True, ax=ax, color=colors)
        else:
            df_st.index = [i.capitalize() for i in df_st.index]
            df_st.plot(kind="bar", stacked=True, ax=ax, color=colors)

        if xlabel == "Year":
            pattern = re.compile(r"\b\d{4}\b")
            # Extract the year from each string in the list
            xlabels = list(df_st.index)
            years = [
                pattern.search(date_string).group()
                for date_string in xlabels
                if pattern.search(date_string)
            ]
            pattern = re.compile(r"\(.*?\)")
            var_name = pattern.sub("", xlabels[0])

            ax.set_xticks(ticks=np.arange(len(xlabels)), labels=years)
            ax.set_xlim([0, len(xlabels) - 1])
            ax.set_ylim([0, 100])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(f"Relative variance contribution\non {var_name} (%)")
            ax.legend(
                title="Uncertainty sources",
                bbox_to_anchor=(1.01, 0.5),
                loc="center left",
                frameon=False,
            )
        else:
            var_name = "peroids"
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Relative variance\ncontribution (%)")
            # ax.tick_params(axis='x', labelrotation=60)
            ax.legend(
                title="Uncertainty sources",
                bbox_to_anchor=(1.01, 0.3),
                loc="center left",
                frameon=False,
            )

        plt.tight_layout()

        if fig_name is not None:
            if fig_name == "":
                fig_name = f"task3_sobel_st_{var_name}_{xlabel}.jpg"
            self.save_figure(fig, fig_name)
        else:
            plt.show()

    def plot_sobo_total_effect_indice_with_un_factor(
        self, df_st_list: list, n_std: int = 2, fig_name: str | None = None
    ):
        """Plot the total effect indices with uncertainty factors.
        Only allow time series data."""

        un_source_colors = {
            "set": self.colors["corn"],
            "prec_id": self.colors["prec"],
            "eco_id": self.colors["crop_price"],
            "py_seed": self.colors["SD6Model"],
            "SpatialGLM": self.colors["corn"],
            "Precipitation": self.colors["prec"],
            "CropPrice": self.colors["crop_price"],
            "SD6Model": self.colors["SD6Model"],
        }
        labels = {
            "SpatialGLM": "SpatialGLM of crops",
            "Precipitation": "Internal climate variability",
            "CropPrice": "Crop price variation",
        }

        # Calculate relative variance contribution of the total effect indices
        df_st_list = [
            df_st.divide(df_st.sum(axis=1), axis=0) * 100 for df_st in df_st_list
        ]
        df_st_acc_list = [df_st.cumsum(axis=1) for df_st in df_st_list]
        df_st_concat = pd.concat(df_st_acc_list, axis=0)
        df_st_mean = df_st_concat.groupby(level=0).mean()
        df_st_std = df_st_concat.groupby(level=0).std()

        df_st_mean_ub = df_st_mean + df_st_std * n_std
        df_st_mean_lb = df_st_mean - df_st_std * n_std

        # Extract x
        pattern = re.compile(r"\b\d{4}\b")
        x = [
            int(pattern.search(date_string).group())
            for date_string in list(df_st_mean.index)
            if pattern.search(date_string)
        ]
        pattern = re.compile(r"\(.*?\)")
        var_name = pattern.sub("", df_st_mean.index[0])
        if "GW_st" in var_name:
            var_name = "Saturated thickness"
            file_var_name = "saturated_thickness"
        if "withdrawal" in var_name:
            var_name = "Withdrawal"
            file_var_name = "withdrawal"

        fontsize = 12
        plt.rcParams["font.size"] = fontsize

        fig, ax = plt.subplots(figsize=(self.figwd["2"], self.figwd["1"] * 1.1))
        num_vars = df_st_mean.shape[1]
        for i in range(num_vars):
            if i == 0:
                y1 = np.zeros(df_st_mean.shape[0])
            else:
                y1 = df_st_mean.iloc[:, i - 1]
            y2 = df_st_mean.iloc[:, i]
            # Fill the area between the two lines
            un_source_name = df_st_mean.columns[i]
            ax.fill_between(
                x,
                y1,
                y2,
                color=un_source_colors[un_source_name],
                label=labels[un_source_name],
            )

        # Add uncertainty
        for i in range(num_vars - 1):
            y1 = df_st_mean_ub.iloc[:, i]
            y2 = df_st_mean_lb.iloc[:, i]
            ax.fill_between(x, y1, y2, color="white", alpha=0.3)

        ax.set_xlabel("Year")
        ax.set_ylabel(f"Relative variance contribution\non {var_name.lower()} (%)")

        ax.set_xlim([x[0], x[-1]])
        ax.set_ylim([0, 100])

        ax.set_xticks(ticks=x)

        ax.legend(
            title="Uncertainty sources",
            bbox_to_anchor=(1.01, 0.5),
            loc="center left",
            frameon=False,
        )

        plt.tight_layout()

        if fig_name is not None:
            if fig_name == "":
                fig_name = f"task3_sobel_st_{file_var_name}_with_un.jpg"
            self.save_figure(fig, fig_name)
        else:
            plt.show()

    def plot_sns_std_barplot_for_task1(
        self,
        df_metrices_std: pd.DataFrame,
        var_: str = "withdrawal",
        hue_order=["LEMA", "LEMA1", "LEMA2", "Dry", "Wet"],
        fig_name: str | None = None,
    ):
        fontsize = 10
        plt.rcParams["font.size"] = fontsize

        data = df_metrices_std[df_metrices_std["variable"] == var_]
        data = data[data["period"].isin(hue_order)]

        fig, ax = plt.subplots(figsize=(self.figwd["1.5"], self.figwd["1"] * 1.1))
        sns.barplot(
            data=data,
            x="Factor",
            y="value",
            hue="period",
            hue_order=hue_order,
            palette="muted",
            ax=ax,
        )

        # Add data points on top of the bars
        sns.stripplot(
            data=data,
            x="Factor",
            y="value",
            hue="period",
            hue_order=hue_order,
            palette="muted",
            dodge=True,  # To separate the points for different hues
            ax=ax,
            marker="o",  # Shape of the data points
            size=5,  # Size of the data points
            linewidth=0.5,  # Edge width of the data points
            edgecolor="gray",  # Edge color of the data points
        )

        # Adjust the legend to avoid duplication
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles[0 : len(handles) // 2],
            labels[0 : len(labels) // 2],
            frameon=False,
            ncol=len(hue_order),
            loc="upper left",
            bbox_to_anchor=(-0.03, 1.12),
            handlelength=1.5,
            handletextpad=0.5,
        )

        if var_ == "withdrawal":
            ax.set_ylabel(f"Standard deviation of {var_} ($10^6$ m$^3$)")
        elif var_ == "GW_st":
            ax.set_ylabel("Standard deviation of saturated thickness (m)")

        if fig_name is not None:
            if fig_name == "":
                fig_name = f"task1_std_barplot_{var_}.jpg"
            self.save_figure(fig, fig_name)
        else:
            plt.show()

    def t1_std_barplot(
        self,
        df_metrices_std: pd.DataFrame,
        var_: str = ["GW_st", "withdrawal"],
        hue_order=["LEMA", "LEMA1", "LEMA2", "Dry", "Wet"],
        fig_name: str | None = None,
    ):
        fontsize = 10
        plt.rcParams["font.size"] = fontsize

        fig, axes = plt.subplots(
            nrows=2, figsize=(self.figwd["1.5"], self.figwd["1"] * 1.1), sharex=True
        )
        axes = axes.flatten()

        figls = ["(a)", "(b)", "(c)", "(d)", "(e)"]

        for i, ax in enumerate(axes):
            data = df_metrices_std[df_metrices_std["variable"] == var_[i]]
            data = data[data["period"].isin(hue_order)]

            sns.barplot(
                data=data,
                x="Factor",
                y="value",
                hue="period",
                hue_order=hue_order,
                palette="muted",
                ax=ax,
            )

            # Add data points on top of the bars
            sns.stripplot(
                data=data,
                x="Factor",
                y="value",
                hue="period",
                hue_order=hue_order,
                palette="muted",
                dodge=True,  # To separate the points for different hues
                ax=ax,
                marker="o",  # Shape of the data points
                size=5,  # Size of the data points
                linewidth=0.5,  # Edge width of the data points
                edgecolor="gray",  # Edge color of the data points
            )

            # Adjust the legend to avoid duplication
            if i == 0:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(
                    handles[0 : len(handles) // 2],
                    labels[0 : len(labels) // 2],
                    frameon=False,
                    ncol=len(hue_order),
                    loc="upper left",
                    bbox_to_anchor=(-0.03, 1.2),
                    handlelength=1.5,
                    handletextpad=0.5,
                )
            else:
                ax.get_legend().remove()

            # Add y labels
            if var_[i] == "withdrawal":
                ax.set_ylabel(
                    f"{figls[i]} Standard deviation\nof withdrawal\n($10^6$ m$^3$)"
                )
            elif var_[i] == "GW_st":
                ax.set_ylabel(
                    f"{figls[i]} Standard deviation\nof saturated thickness\n(m)"
                )
            ax.set_xlabel("")

        # Align y-axis labels across subplots
        fig.align_ylabels()
        if fig_name is not None:
            if fig_name == "":
                fig_name = f"task1_std_barplot.jpg"
            self.save_figure(fig, fig_name)
        else:
            plt.show()

    def t2_lineplots(
        self,
        df_ts: pd.DataFrame,
        fig_name: str | None = None,
        add_LEMA_vlines: bool = True,
        add_dry_year_regions: bool = True,
        add_prec_bars_on_secondary_yaxis: bool = True,
        **kwargs,
    ) -> None:
        fontsize = 12
        plt.rcParams["font.size"] = fontsize
        var_list = ["corn", "withdrawal", "GW_st"]
        ylabel_list = [
            "Corn ratio",
            "Withdrawal ($10^6$ m$^3$)",
            "Saturated thickness (m)",
        ]

        fig, axes = plt.subplots(
            figsize=(self.figwd["2"], self.figwd["2"] * 1),
            nrows=len(var_list),
            sharex=True,
        )
        axes = axes.flatten()
        hue_order = sorted(df_ts["ratio_group"].unique())
        style_order = sorted(df_ts["range"].unique())
        for i, var_name in enumerate(var_list):
            ax = axes[i]
            sns.lineplot(
                data=df_ts,
                x="year",
                y=var_name,
                hue="ratio_group",
                style="range",
                hue_order=hue_order,
                style_order=style_order,
                ax=ax,
                palette="YlOrBr",
                **kwargs,
            )
            if add_LEMA_vlines:
                self.add_LEMA_vlines(ax)
            if add_dry_year_regions:
                self.add_dry_year_regions(ax)
            if add_prec_bars_on_secondary_yaxis:
                self.add_prec_bars_on_secondary_yaxis(ax, invert_yaxis=False)
            info = self.sd6_plotting_info
            ax.set_xlim(info.year_range)
            self.auto_ylim(ylabel_list[i], ax)
            ax.set_xlabel("Year")
            ax.set_ylabel(ylabel_list[i])
            if var_name == "withdrawal":
                ax.set_ylim(8, 42)
            # Create custom legend
            if i == 0:
                handles, labels = ax.get_legend_handles_labels()
                new_handles = [
                    Line2D(
                        [0],
                        [0],
                        color=handle.get_color(),
                        linestyle=handle.get_linestyle(),
                        lw=3,
                    )
                    if isinstance(handle, Line2D)
                    else handle
                    for handle in handles
                ]
                # Assuming the first two handles and labels are for 'ratio_group' and the rest are for 'range'
                new_handles = [
                    Line2D([0], [0], color=handle.get_color(), lw=3)
                    for handle in handles[1 : len(hue_order) + 1]
                ]
                legend_ratio = ax.legend(
                    handles=new_handles,
                    labels=labels[1 : len(hue_order) + 1],
                    title="Corn ratio\ngroup (%)",
                    bbox_to_anchor=(1.07, -0.85),
                    loc="center left",
                    borderaxespad=0.0,
                    ncol=1,
                    frameon=False,
                )
                ax.add_artist(legend_ratio)

                linestyles = [
                    "-",
                    "-.",
                    ":",
                    "--",
                    (0, (3, 5, 1, 5, 1, 5)),
                    (0, (1, 5)),
                    (0, (5, 5, 1, 5)),
                ]
                new_handles = [
                    Line2D([0], [0], linestyle=ls, color="k", lw=3)
                    for handle, ls in zip(handles[len(hue_order) + 2 :], linestyles)
                ]
                legend_range = ax.legend(
                    handles=new_handles,
                    labels=labels[len(hue_order) + 2 :],
                    title="Range (km)",
                    bbox_to_anchor=(1.28, -0.33),
                    loc="center left",
                    handlelength=5,
                    borderaxespad=0.0,
                    ncol=1,
                    frameon=False,
                )
                ax.add_artist(legend_range)

                blue_bar_patch = Patch(
                    color=self.colors["prec"], alpha=0.5, label="Annual\nprecipitation"
                )
                gray_patch = Patch(color=self.colors["dry_year"], label="Dry years")
                ax.legend(
                    handles=[blue_bar_patch, gray_patch],
                    bbox_to_anchor=(1.28, -1.5),
                    loc="center left",
                    borderaxespad=0.0,
                    ncol=1,
                    frameon=False,
                )

            else:
                ax.get_legend().remove()

        # Show the plot
        # plt.tight_layout()  # Adjust the layout to make room for the legend

        if fig_name is not None:
            if fig_name == "":
                fig_name = f"task2_ts.jpg"
            self.save_figure(fig, fig_name)
        else:
            plt.show()

    def appendix_plot_B_regr_bounds(self, fig_name: str | None = None):
        def func(st, k, sy):
            # constants
            r = 0.4064 / 2
            d = 90
            eps = 0.5

            # transmissivity
            T = k * st
            ans = (
                1
                / (4 * np.pi * T * eps)
                * (-0.5772 - np.log(r**2 * sy / (4 * T * d)))
            )
            return ans

        # %% st vs B
        st_list = np.arange(14, 22 + 0.05, 0.05)

        min_B = [func(st, 117.85, 0.04) for st in st_list]
        max_B = [func(st, 45.34, 0.082) for st in st_list]

        fontsize = 11
        plt.rcParams["font.size"] = fontsize
        fig, ax = plt.subplots(figsize=(self.figwd["2"], self.figwd["1"] * 0.92))

        # Perform linear regression (numpy.polyfit can be used for this purpose)
        slope_min, intercept_min = np.polyfit(st_list, min_B, 1)
        slope_max, intercept_max = np.polyfit(st_list, max_B, 1)

        # Generate y-values for regression lines
        regression_line_min = np.polyval([slope_min, intercept_min], st_list)
        regression_line_max = np.polyval([slope_max, intercept_max], st_list)

        ax.plot(st_list, min_B, label="Min B in the SD-6 LEMA", color="blue")
        ax.plot(
            st_list,
            regression_line_min,
            color="deepskyblue",
            ls="--",
            label=f"y = {slope_min:.4f}x + {intercept_min:.4f}",
        )
        ax.plot(st_list, max_B, label="Max B in the SD-6 LEMA", color="green")
        ax.plot(
            st_list,
            regression_line_max,
            "y--",
            label=f"y = {slope_max:.4f}x + {intercept_max:.4f}",
        )

        ax.legend(loc="center", bbox_to_anchor=(1.45, 0.5), ncol=1, frameon=False)
        ax.set_xlabel("Saturated thickness (m); x")
        ax.set_ylabel("$B; y$")
        # ax.set_ylabel("$B = 1/(4\pi T\epsilon) * [-0.5772 - ln(r^2 sy/(4Td))]$")

        plt.tight_layout()

        if fig_name is not None:
            if fig_name == "":
                fig_name = "Appendix_B_regr_bounds.jpg"
            self.save_figure(fig, fig_name)
        else:
            plt.show()

    def appendix_plot_k_sy_B(self, PDIV_Well_Info_path, fig_name: str | None = None):
        def func(st, k, sy):
            # constants
            r = 0.4064 / 2
            d = 90
            eps = 0.5

            # transmissivity
            T = k * st
            ans = (
                1
                / (4 * np.pi * T * eps)
                * (-0.5772 - np.log(r**2 * sy / (4 * T * d)))
            )
            return ans

        df = pd.read_csv(PDIV_Well_Info_path)
        # Parameters
        st = 18  # fixed st value
        k_list = np.arange(45.34, 117.85 + 1, 1)
        sy_list = np.arange(0.04, 0.082 + 0.01, 0.01)

        # Prepare the matrix to hold the function outputs
        values = np.zeros((len(k_list), len(sy_list)))

        # Calculate values for each combination of k and sy
        for i, k in enumerate(k_list):
            for j, sy in enumerate(sy_list):
                values[i, j] = func(st, k, sy)

        # Create a heatmap
        fig, ax = plt.subplots(figsize=(self.figwd["1.5"], self.figwd["1"]))
        c = ax.imshow(
            values,
            interpolation="nearest",
            origin="lower",
            extent=[k_list.min(), k_list.max(), sy_list.min(), sy_list.max()],
            aspect="auto",
        )
        cbar = fig.colorbar(c, ax=ax)
        cbar.set_label("$B$ | saturated thickness = 18 m")
        ax.scatter(
            df["hydraulic_conductivity_mday"],
            df["specific_yield"],
            c="white",
            marker="x",
            s=10,
        )

        # Label axes
        ax.set_xlabel("Hydrulic conductivity (m/day)")
        ax.set_ylabel("Specific yield")
        # ax.set_title("$B = 1/(4\pi T\epsilon) * [-0.5772 - ln(r^2 sy/(4Td))]$ with st = 18")

        if fig_name is not None:
            if fig_name == "":
                fig_name = "Appendix_k-sy_B.jpg"
            self.save_figure(fig, fig_name)
        else:
            plt.show()

    def plot_sns_heatmap_with_prec_eco_uncertainty(
        self,
        pilot_case: tuple,
        other_points: pd.DataFrame,
        base_heatmap: pd.DataFrame,
        do_row_normalization: bool = False,
        palette: str = "viridis",
        fig_name: str = None,
        legend_title: str = "Withdrawal $10^6$ m$^3$",
        xlabel: str | None = "Range (km)",
        ylabel: str | None = "Corn ratio group (%)",
        **kwargs,
    ) -> None:
        table = base_heatmap

        fontsize = 9
        plt.rcParams["font.size"] = fontsize
        fig, ax = plt.subplots(figsize=(self.figwd["1"], self.figwd["1"] * 0.9))

        # Create a colormap that sets NaN values to grey
        cmap = sns.color_palette(palette, as_cmap=True)
        cmap.set_bad("grey")  # 'bad' data points (NaNs) are grey
        cbar_kws = {"extend": "neither", "label": legend_title}

        # Create the heatmap (base map)
        if not do_row_normalization:
            sns.heatmap(table, cmap=cmap, cbar_kws=cbar_kws, ax=ax, zorder=1, **kwargs)
            row_type = "standard"
        else:
            table_row_normalized = table.sub(table.mean(axis=1), axis=0)
            sns.heatmap(
                table_row_normalized,
                cmap=cmap,
                cbar_kws=cbar_kws,
                ax=ax,
                zorder=1,
                **kwargs,
            )
            row_type = "row_normalized"

        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        # Add the pilot case and other points
        ax.scatter(
            pilot_case[0],
            pilot_case[1],
            s=20,
            c="yellow",
            marker="*",
            edgecolors="black",
            zorder=5,
        )

        ax.scatter(
            other_points["range"],
            other_points["corn_ratio"],
            s=10,
            c="white",
            marker="x",
            zorder=4,
        )

        if fig_name is not None:
            if fig_name == "":
                fig_name = f"task4_heatmap.jpg"
            self.save_figure(fig, fig_name)
        else:
            plt.show()

import pandas as pd
import pathnavigator
import clt

root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\SD6InternalVariability"
pn = pathnavigator.create(root_dir)
pn.code.chdir()

from anova_utils import plot_anova_sum_sq_fraction, plot_norm_comparison

# The plotting data is copied from outputs
mu_dict = clt.io.read_pd_hdf5(pn.figures.data_for_plotting.get()/"anova_mu_fraction_withIV_seperated.h5")
plot_anova_sum_sq_fraction(mu_dict, pn.figures.get() / "fig4_anova.jpg")


df_Wi_regime_norm = pd.read_csv(pn.figures.data_for_plotting.get()/"df_Wi_regime_norm.csv", index_col=[0])
plot_norm_comparison(pn, df_Wi_regime_norm, save_figname=pn.figures.get() / "fig5_anova_irrigation_norm.jpg")



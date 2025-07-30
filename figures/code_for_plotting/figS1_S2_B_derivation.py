import pathnavigator

root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\SD6InternalVariability"
pn = pathnavigator.create(root_dir)
pn.code.chdir()

from plotting import SD6Visual

visual = SD6Visual(output_dir=pn.figures.get())

visual.appendix_plot_k_sy_B(
    PDIV_Well_Info_path=pn.data.get()/"PDIV_Well_Info.csv",
    fig_name="figS1_k-sy_B.jpg"
    )

visual.appendix_plot_B_regr_bounds(fig_name="figS2_B_regr_bounds.jpg")

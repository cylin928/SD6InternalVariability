import pandas as pd
import matplotlib.pyplot as plt
import pathnavigator

root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\SD6InternalVariability"
pn = pathnavigator.create(root_dir)

# Load the dataset
file_path = pn.figures.data_for_plotting.get("figS4_survey_data.csv")
data = pd.read_csv(file_path)

# Select just the required columns
plant_columns = [col for col in data.columns if col.startswith('plant_') and col not in ['plant_other_text', 'plant_depend']]
plant_averages = data[plant_columns].mean()

# Mapping of original column names to descriptive labels
column_rename_map = {
    'plant_cropprice': 'Expected crop prices',
    'plant_inputprice': 'Expected input prices',
    'plant_labortime': 'Labor and time requirement',
    'plant_weather': 'Expected weather',
    'plant_surfacew': 'Forcasted surface water',
    'plant_wlevel': 'Water table level',
    'plant_knowledge': 'Knowledge of / experience with crop',
    'plant_sustainability': 'Environmental sustainability',
    'plant_equipment': 'Crop-specific equipment',
    'plant_risk': 'Risk/uncertainty of crop',
    'plant_buyer': 'Known buyer',
    'plant_neighbor': "Neighbor's decision",
    'plant_rotation': 'Set rotation',
    'plant_soil': 'Soil characters',
    'plant_other': 'Other'
}

# Replace index (column names) in the plant_averages Series
plant_averages.rename(index=column_rename_map, inplace=True)

# Customization options
label_fontsize = 12
tick_fontsize = 12
bar_color = 'skyblue'

# Create horizontal bar chart with sorted values (highest at top)
plt.figure(figsize=(7, 4),dpi = 300)
plant_averages.sort_values(ascending=True).plot(kind='barh', color=bar_color)

# Customize plot
plt.xlabel('Average rating of importance\n(higher is more important)', fontsize=label_fontsize)
plt.ylabel('Factor', fontsize=label_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Save the plot to a file
plt.tight_layout()
plt.savefig(pn.figures.get() / "figS4_survey_data.jpg", dpi=300)

# Show the plot
plt.show()



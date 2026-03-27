import os
import pandas as pd

# folder where all individual model result CSVs are saved
RESULTS_FOLDER = "../results"

# output excel file name
OUTPUT_FILE = os.path.join(RESULTS_FOLDER, "all_model_comparisons_real_world.xlsx")

all_dfs = []

# loop through every csv in results folder
for file_name in os.listdir(RESULTS_FOLDER):
    if file_name.endswith("world.csv"):
        file_path = os.path.join(RESULTS_FOLDER, file_name)
        df = pd.read_csv(file_path)
        all_dfs.append(df)

# combine all rows into one dataframe
final_df = pd.concat(all_dfs, ignore_index=True)

# optional: sort by model name
final_df = final_df.sort_values(by="Model").reset_index(drop=True)

# save as excel
final_df.to_excel(OUTPUT_FILE, index=False)

print("Combined Excel file created successfully!")
print("Saved at:", OUTPUT_FILE)
print("\nPreview:")
print(final_df)
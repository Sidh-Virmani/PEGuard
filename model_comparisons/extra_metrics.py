import os
import pandas as pd

RESULTS_FOLDER = "../results"
OUTPUT_CSV = os.path.join(RESULTS_FOLDER, "all_models_with_extra_metrics_real_world.csv")
OUTPUT_EXCEL = os.path.join(RESULTS_FOLDER, "all_models_with_extra_metrics_real_world.xlsx")

all_dfs = []

for file_name in os.listdir(RESULTS_FOLDER):
    # only read the individual model result csv files
    if file_name.endswith("world.csv") and not file_name.startswith("all_"):
        file_path = os.path.join(RESULTS_FOLDER, file_name)
        df = pd.read_csv(file_path)
        all_dfs.append(df)

# combine all model rows
final_df = pd.concat(all_dfs, ignore_index=True)

# -----------------------------
# Add new metrics
# -----------------------------

# 1. Specificity = TN / (TN + FP)
final_df["Specificity"] = final_df["TN"] / (final_df["TN"] + final_df["FP"])

# 2. Balanced Accuracy = (Recall + Specificity) / 2
final_df["Balanced Accuracy"] = (final_df["Recall"] + final_df["Specificity"]) / 2

# optional: round for neat display
cols_to_round = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC",
    "False Positive Rate",
    "False Negative Rate",
    "Specificity",
    "Balanced Accuracy"
]

for col in cols_to_round:
    if col in final_df.columns:
        final_df[col] = final_df[col].round(6)

# sort by model name
final_df = final_df.sort_values(by="Model").reset_index(drop=True)

# save
final_df.to_csv(OUTPUT_CSV, index=False)
final_df.to_excel(OUTPUT_EXCEL, index=False)

print("Done!")
print("Saved CSV to:", OUTPUT_CSV)
print("Saved Excel to:", OUTPUT_EXCEL)
print("\nPreview:")
print(final_df)
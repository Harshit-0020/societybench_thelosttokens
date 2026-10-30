import pandas as pd

# Save the dataset as the Excel file your script expects
csv_file = "articles.csv"  # or your CSV filename
excel_file = "russia_ukraine_war_developments_2024.xlsx"

df = pd.read_csv(csv_file)
df.to_excel(excel_file, index=False)

print(f"Successfully created '{excel_file}' for dataset_builder.py")
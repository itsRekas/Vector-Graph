import pandas as pd

# Read the file line by line into a pandas DataFrame
with open('RLUBM.nt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Create a DataFrame from the lines
df = pd.DataFrame({'line': lines})

# Remove duplicate lines
df_cleaned = df.drop_duplicates()

# Write to the new file, preserving exact format
with open('RLUBM_cleaned.nt', 'w', encoding='utf-8') as f:
    f.writelines(df_cleaned['line'].tolist())

print(f"Original lines: {len(df)}")
print(f"Cleaned lines: {len(df_cleaned)}")
print(f"Removed {len(df) - len(df_cleaned)} duplicate lines")
print("Cleaned file saved as RLUBM_cleaned.nt")


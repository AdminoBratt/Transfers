import pandas as pd
import glob


csv_files = glob.glob("*.csv")
print(f"Found {len(csv_files)} CSV files: {csv_files}")


combined_df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)


combined_df.to_csv("combined_transfers.csv", index=False)
print(f"Combined CSV saved as 'combined_transfers.csv' with {len(combined_df)} rows")

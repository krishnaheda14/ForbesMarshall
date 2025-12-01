import pandas as pd
import os

# Define paths relative to the script location
base_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(base_dir, 'data', 'jobs_dataset.csv')

try:
    print(f"Reading from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows.")

    # Count bad deadlines before fix
    bad_count = len(df[df['Due_Day'] < df['Release_Day']])
    print(f"Found {bad_count} operations with impossible deadlines.")

    # Apply Logic: If Due Date < Release Date, treat Due Date as Lead Time (Release + Due)
    def fix_date(row):
        if row['Due_Day'] < row['Release_Day']:
            return row['Release_Day'] + row['Due_Day']
        return row['Due_Day']

    df['Due_Day'] = df.apply(fix_date, axis=1)

    # Save back to the same file
    df.to_csv(input_file, index=False)
    
    print("✅ Success! File updated.")
    print("Sample of corrected data:")
    print(df[['Job_ID', 'Release_Day', 'Due_Day']].head())

except FileNotFoundError:
    print(f"❌ Error: File not found at {input_file}")
except Exception as e:
    print(f"❌ An error occurred: {e}")
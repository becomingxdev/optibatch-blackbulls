import pandas as pd
from config import PRODUCTION_FILE, PROCESS_FILE

def load_production_data():
    df = pd.read_excel(PRODUCTION_FILE)
    print(f"Production data loaded: {df.shape}")
    return df

def load_process_data():
    xl = pd.ExcelFile(PROCESS_FILE)
    all_batches = []

    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        df["source_sheet"] = sheet
        all_batches.append(df)

    process_df = pd.concat(all_batches, ignore_index=True)
    print(f"Process data loaded: {process_df.shape}")
    return process_df